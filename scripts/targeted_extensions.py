from __future__ import annotations

from config import OUTPUT_DIR as OUT

import numpy as np
import pandas as pd
import pyhdfe
import statsmodels.api as sm
from scipy.stats import chi2, norm


import multidomain_analysis as upgrade
locked = upgrade.locked
base = upgrade.base

EVENT_TERMS = locked.EVENT_TERMS
POST_TERMS = ["post1", "post2"]
AGE_GRID = list(range(50, 91))
CONTEXT_MODIFIERS = {
    "high_school_plus": "Higher education",
    "married_partnered": "Married or partnered",
    "living_alone": "Living alone",
}


def add_context(site: str, panel: pd.DataFrame) -> pd.DataFrame:
    if site == "CHARLS":
        path, id_col, hh_col = locked.CHARLS, "ID", "h1hhres"
        education_col, marital_col = "raeducl", "r1mstat"
    else:
        path, id_col, hh_col = base.HRS, "hhidpn", "h10hhres"
        education_col, marital_col = "raeduc", "r10mstat"
    raw = pd.read_stata(
        path,
        columns=[id_col, hh_col, education_col, marital_col],
        convert_categoricals=False,
        preserve_dtypes=False,
    )
    if site == "CHARLS":
        raw["ID"] = raw[id_col].astype(str)
    else:
        raw["ID"] = raw[id_col].round().astype("Int64").astype(str)
    raw["household_size"] = pd.to_numeric(raw[hh_col], errors="coerce").where(
        lambda x: x.between(1, 20)
    )
    raw["living_alone"] = np.where(
        raw["household_size"].notna(), raw["household_size"].eq(1).astype(float), np.nan
    )
    if site == "CHARLS":
        raw["high_school_plus_context"] = np.where(
            raw[education_col].notna(), raw[education_col].ge(2).astype(float), np.nan
        )
        raw["married_partnered_context"] = np.where(
            raw[marital_col].notna(), raw[marital_col].isin([1, 3]).astype(float), np.nan
        )
    else:
        raw["high_school_plus_context"] = np.where(
            raw[education_col].notna(), raw[education_col].ge(3).astype(float), np.nan
        )
        raw["married_partnered_context"] = np.where(
            raw[marital_col].notna(), raw[marital_col].isin([1, 2, 3]).astype(float), np.nan
        )
    raw = raw[[
        "ID", "household_size", "living_alone",
        "high_school_plus_context", "married_partnered_context",
    ]].drop_duplicates("ID")
    out = panel.merge(raw, on="ID", how="left", validate="many_to_one")
    out["high_school_plus"] = out["high_school_plus_context"]
    out["married_partnered"] = out["married_partnered_context"]
    return out.drop(columns=["high_school_plus_context", "married_partnered_context"])


def fit_design(stack: pd.DataFrame, outcome: str, design_terms: list[str]):
    z = stack[stack["inw"].eq(1) & stack[outcome].notna()].copy()
    required = sorted(set(design_terms) - set(EVENT_TERMS))
    if required:
        z = z.dropna(subset=required)
    panel_n = z.groupby("person_fe").size()
    z = z[z["person_fe"].isin(panel_n[panel_n.ge(2)].index)].copy()
    ids = np.column_stack([
        pd.factorize(z["person_fe"])[0],
        pd.factorize(z["wave_fe"])[0],
    ])
    algorithm = pyhdfe.create(ids, drop_singletons=False, residualize_method="map")
    y = algorithm.residualize(z[[outcome]].to_numpy(float)).ravel()
    x = algorithm.residualize(z[design_terms].to_numpy(float))
    keep = np.nanstd(x, axis=0) > 1e-12
    names = np.asarray(design_terms)[keep].tolist()
    fit = sm.OLS(y, x[:, keep]).fit(
        cov_type="cluster", cov_kwds={"groups": z["ID"]}
    )
    return fit, names, z


def named_vector(names: list[str], values: dict[str, float]) -> np.ndarray:
    return np.asarray([values.get(name, 0.0) for name in names], dtype=float)


def linear_age_model(site: str, stack: pd.DataFrame, age_col: str, sd: float) -> list[dict]:
    z = stack.copy()
    z["age_per_10y"] = (pd.to_numeric(z[age_col], errors="coerce") - 65.0) / 10.0
    interaction_terms = []
    for term in EVENT_TERMS:
        name = f"{term}_x_age_per_10y"
        z[name] = z[term] * z["age_per_10y"]
        interaction_terms.append(name)
    fit, names, used = fit_design(z, "adl5", [*EVENT_TERMS, *interaction_terms])
    rows = []
    for term in EVENT_TERMS:
        name = f"{term}_x_age_per_10y"
        if name not in names:
            continue
        j = names.index(name)
        beta, se = float(fit.params[j]), float(fit.bse[j])
        rows.append({
            "cohort": site,
            "term": term,
            "n_unique_people": used["ID"].nunique(),
            "age_interaction_per_10y": beta,
            "std_error": se,
            "ci_low": beta - 1.96 * se,
            "ci_high": beta + 1.96 * se,
            "p_interaction": float(fit.pvalues[j]),
            "standardized_age_interaction_per_10y": beta / sd,
            "standardized_std_error": se / sd,
            "standardized_ci_low": (beta - 1.96 * se) / sd,
            "standardized_ci_high": (beta + 1.96 * se) / sd,
        })
    return rows


def rcs_component(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    k1, k2, k3 = knots
    pos = lambda value: np.maximum(value, 0.0) ** 3
    return (
        (pos(x - k1) - pos(x - k3)) / (k3 - k1)
        - (pos(x - k2) - pos(x - k3)) / (k3 - k2)
    )


def spline_age_model(site: str, stack: pd.DataFrame, age_col: str, sd: float) -> tuple[list[dict], list[dict]]:
    z = stack.copy()
    ages = pd.to_numeric(z[age_col], errors="coerce")
    unique_ages = pd.to_numeric(
        z[["ID", age_col]].drop_duplicates("ID")[age_col], errors="coerce"
    ).dropna()
    knots = unique_ages.quantile([0.10, 0.50, 0.90]).to_numpy(float)
    z["age_per_10y"] = (ages - 65.0) / 10.0
    raw_rcs = rcs_component(ages.to_numpy(float), knots)
    center_rcs = float(rcs_component(np.asarray([65.0]), knots)[0])
    scale_rcs = float(np.nanstd(raw_rcs))
    z["age_rcs_nonlin"] = (raw_rcs - center_rcs) / scale_rcs
    design_terms = list(EVENT_TERMS)
    for term in EVENT_TERMS:
        for component in ["age_per_10y", "age_rcs_nonlin"]:
            name = f"{term}_x_{component}"
            z[name] = z[term] * z[component]
            design_terms.append(name)
    fit, names, used = fit_design(z, "adl5", design_terms)
    cov = np.asarray(fit.cov_params())
    prediction_rows, test_rows = [], []
    for term in EVENT_TERMS:
        nonlinear_name = f"{term}_x_age_rcs_nonlin"
        if nonlinear_name in names:
            j = names.index(nonlinear_name)
            test_rows.append({
                "cohort": site,
                "term": term,
                "n_unique_people": used["ID"].nunique(),
                "knots_years": ";".join(f"{k:.1f}" for k in knots),
                "p_nonlinearity": float(fit.pvalues[j]),
            })
        for age in AGE_GRID:
            age10 = (age - 65.0) / 10.0
            h = (float(rcs_component(np.asarray([age]), knots)[0]) - center_rcs) / scale_rcs
            vector = named_vector(names, {
                term: 1.0,
                f"{term}_x_age_per_10y": age10,
                f"{term}_x_age_rcs_nonlin": h,
            })
            estimate = float(vector @ fit.params)
            se = float(np.sqrt(max(0.0, vector @ cov @ vector)))
            prediction_rows.append({
                "cohort": site,
                "term": term,
                "age_years": age,
                "n_unique_people": used["ID"].nunique(),
                "estimate": estimate,
                "std_error": se,
                "ci_low": estimate - 1.96 * se,
                "ci_high": estimate + 1.96 * se,
                "standardized_estimate": estimate / sd,
                "standardized_std_error": se / sd,
                "standardized_ci_low": (estimate - 1.96 * se) / sd,
                "standardized_ci_high": (estimate + 1.96 * se) / sd,
            })
    return prediction_rows, test_rows


def contextual_iadl_model(
    site: str,
    stack: pd.DataFrame,
    modifier: str,
    label: str,
    sd: float,
) -> tuple[list[dict], list[dict]]:
    z = stack.copy()
    interactions = []
    for term in EVENT_TERMS:
        name = f"{term}_x_{modifier}"
        z[name] = z[term] * z[modifier]
        interactions.append(name)
    fit, names, used = fit_design(z, "iadl4", [*EVENT_TERMS, *interactions])
    cov = np.asarray(fit.cov_params())
    interaction_rows, stratum_rows = [], []
    for term in EVENT_TERMS:
        interaction_name = f"{term}_x_{modifier}"
        if interaction_name not in names:
            continue
        j = names.index(interaction_name)
        beta, se = float(fit.params[j]), float(fit.bse[j])
        interaction_rows.append({
            "cohort": site,
            "modifier": modifier,
            "modifier_label": label,
            "term": term,
            "n_unique_people": used["ID"].nunique(),
            "interaction_estimate": beta,
            "std_error": se,
            "ci_low": beta - 1.96 * se,
            "ci_high": beta + 1.96 * se,
            "p_interaction": float(fit.pvalues[j]),
            "standardized_interaction_estimate": beta / sd,
            "standardized_std_error": se / sd,
            "standardized_ci_low": (beta - 1.96 * se) / sd,
            "standardized_ci_high": (beta + 1.96 * se) / sd,
        })
        for value in [0, 1]:
            vector = named_vector(names, {term: 1.0, interaction_name: float(value)})
            estimate = float(vector @ fit.params)
            pred_se = float(np.sqrt(max(0.0, vector @ cov @ vector)))
            stratum_rows.append({
                "cohort": site,
                "modifier": modifier,
                "modifier_label": label,
                "modifier_value": value,
                "term": term,
                "n_unique_people": used["ID"].nunique(),
                "estimate": estimate,
                "std_error": pred_se,
                "ci_low": estimate - 1.96 * pred_se,
                "ci_high": estimate + 1.96 * pred_se,
                "standardized_estimate": estimate / sd,
                "standardized_std_error": pred_se / sd,
                "standardized_ci_low": (estimate - 1.96 * pred_se) / sd,
                "standardized_ci_high": (estimate + 1.96 * pred_se) / sd,
            })
    return interaction_rows, stratum_rows


def fixed_pool_interactions(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (modifier, label, term), z in data.groupby(["modifier", "modifier_label", "term"]):
        if set(z["cohort"]) != {"CHARLS", "HRS"}:
            continue
        yi = z["standardized_interaction_estimate"].to_numpy(float)
        vi = z["standardized_std_error"].to_numpy(float) ** 2
        wi = 1.0 / vi
        estimate = float(np.sum(wi * yi) / np.sum(wi))
        se = float(np.sqrt(1.0 / np.sum(wi)))
        q = float(np.sum(wi * (yi - estimate) ** 2))
        rows.append({
            "cohort": "Fixed-effect pooled",
            "modifier": modifier,
            "modifier_label": label,
            "term": term,
            "standardized_interaction_estimate": estimate,
            "standardized_std_error": se,
            "standardized_ci_low": estimate - 1.96 * se,
            "standardized_ci_high": estimate + 1.96 * se,
            "p_interaction": 2 * norm.sf(abs(estimate / se)),
            "Q": q,
            "Q_p_value": chi2.sf(q, 1),
        })
    pooled = pd.DataFrame(rows)
    pooled["q_value_bh"] = np.nan
    idx = pooled["term"].isin(POST_TERMS)
    pooled.loc[idx, "q_value_bh"] = base.bh_adjust(pooled.loc[idx, "p_interaction"])
    return pooled


def cross_cohort_strata(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, z in data.groupby(["modifier", "modifier_label", "modifier_value", "term"]):
        if set(z["cohort"]) != {"CHARLS", "HRS"}:
            continue
        modifier, label, value, term = keys
        charls = z[z["cohort"].eq("CHARLS")].iloc[0]
        hrs = z[z["cohort"].eq("HRS")].iloc[0]
        difference = hrs.standardized_estimate - charls.standardized_estimate
        se = np.sqrt(hrs.standardized_std_error**2 + charls.standardized_std_error**2)
        rows.append({
            "modifier": modifier,
            "modifier_label": label,
            "modifier_value": value,
            "term": term,
            "CHARLS_standardized_estimate": charls.standardized_estimate,
            "HRS_standardized_estimate": hrs.standardized_estimate,
            "HRS_minus_CHARLS_standardized": difference,
            "std_error": se,
            "ci_low": difference - 1.96 * se,
            "ci_high": difference + 1.96 * se,
            "p_difference": 2 * norm.sf(abs(difference / se)),
        })
    out = pd.DataFrame(rows)
    out["q_difference_bh"] = np.nan
    idx = out["term"].isin(POST_TERMS)
    out.loc[idx, "q_difference_bh"] = base.bh_adjust(out.loc[idx, "p_difference"])
    return out


def cross_cohort_age_slopes(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for term, z in data.groupby("term"):
        if set(z["cohort"]) != {"CHARLS", "HRS"}:
            continue
        charls = z[z["cohort"].eq("CHARLS")].iloc[0]
        hrs = z[z["cohort"].eq("HRS")].iloc[0]
        difference = (
            hrs.standardized_age_interaction_per_10y
            - charls.standardized_age_interaction_per_10y
        )
        se = np.sqrt(charls.standardized_std_error**2 + hrs.standardized_std_error**2)
        rows.append({
            "term": term,
            "HRS_minus_CHARLS_age_interaction_per_10y": difference,
            "std_error": se,
            "ci_low": difference - 1.96 * se,
            "ci_high": difference + 1.96 * se,
            "p_difference": 2 * norm.sf(abs(difference / se)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    _, charls0 = locked.load_charls()
    _, hrs0 = base.load_hrs()
    charls = add_context("CHARLS", upgrade.add_iadl("CHARLS", charls0))
    hrs = add_context("HRS", upgrade.add_iadl("HRS", hrs0))

    age_linear_rows, age_spline_rows, age_nonlinearity_rows = [], [], []
    context_interaction_rows, context_stratum_rows = [], []
    context_audit = []

    for site, panel, cohorts, baseline_wave, age_col in [
        ("CHARLS", charls, locked.CHARLS_EVENT_COHORTS, 1, "r1agey"),
        ("HRS", hrs, locked.HRS_EVENT_COHORTS, 10, "r10agey_b"),
    ]:
        stack, _ = base.make_stack(panel, cohorts, "not_yet")
        adl_sd = float(panel.loc[panel["wave"].eq(baseline_wave), "adl5"].std())
        iadl_sd = float(panel.loc[panel["wave"].eq(baseline_wave), "iadl4"].std())
        age_linear_rows.extend(linear_age_model(site, stack, age_col, adl_sd))
        predictions, nonlinear = spline_age_model(site, stack, age_col, adl_sd)
        age_spline_rows.extend(predictions)
        age_nonlinearity_rows.extend(nonlinear)
        baseline = panel[panel["wave"].eq(baseline_wave)].drop_duplicates("ID")
        for modifier, label in CONTEXT_MODIFIERS.items():
            interactions, strata = contextual_iadl_model(
                site, stack, modifier, label, iadl_sd
            )
            context_interaction_rows.extend(interactions)
            context_stratum_rows.extend(strata)
            observed = baseline[modifier].notna()
            context_audit.append({
                "cohort": site,
                "modifier": modifier,
                "modifier_label": label,
                "eligible_baseline_n": baseline["ID"].nunique(),
                "observed_n": int(observed.sum()),
                "observed_pct": 100 * observed.mean(),
                "modifier_positive_n": int(baseline.loc[observed, modifier].eq(1).sum()),
                "modifier_positive_pct": 100 * baseline.loc[observed, modifier].eq(1).mean(),
            })

    age_linear = pd.DataFrame(age_linear_rows)
    age_spline = pd.DataFrame(age_spline_rows)
    age_nonlinearity = pd.DataFrame(age_nonlinearity_rows)
    context_interactions = pd.DataFrame(context_interaction_rows)
    context_strata = pd.DataFrame(context_stratum_rows)
    pooled_context = fixed_pool_interactions(context_interactions)
    context_interactions = pd.concat(
        [context_interactions, pooled_context], ignore_index=True, sort=False
    )
    context_differences = cross_cohort_strata(context_strata)

    age_linear["q_value_bh"] = np.nan
    idx = age_linear["term"].isin(POST_TERMS)
    age_linear.loc[idx, "q_value_bh"] = base.bh_adjust(age_linear.loc[idx, "p_interaction"])
    age_slope_differences = cross_cohort_age_slopes(age_linear)

    age_linear.to_csv(OUT / "Table_S10_ADL_age_linear_interactions.csv", index=False, encoding="utf-8-sig")
    age_slope_differences.to_csv(OUT / "Table_S10b_ADL_age_interaction_cross_cohort.csv", index=False, encoding="utf-8-sig")
    age_spline.to_csv(OUT / "Table_S11_ADL_age_spline_predictions.csv", index=False, encoding="utf-8-sig")
    age_nonlinearity.to_csv(OUT / "Table_S11b_ADL_age_spline_nonlinearity.csv", index=False, encoding="utf-8-sig")
    context_interactions.to_csv(OUT / "Table_S12_IADL_context_interactions.csv", index=False, encoding="utf-8-sig")
    context_strata.to_csv(OUT / "Table_S13_IADL_context_stratum_estimates.csv", index=False, encoding="utf-8-sig")
    context_differences.to_csv(OUT / "Table_S14_IADL_context_cross_cohort_differences.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(context_audit).to_csv(OUT / "Table_S15_context_variable_audit.csv", index=False, encoding="utf-8-sig")
    age_spline[age_spline["term"].isin(POST_TERMS)].to_csv(
        OUT / "Source_Data_Figure_2_age_dose_response.csv", index=False, encoding="utf-8-sig"
    )
    context_interactions[context_interactions["term"].isin(POST_TERMS)].to_csv(
        OUT / "Source_Data_Figure_S7_IADL_context.csv", index=False, encoding="utf-8-sig"
    )

    print("Continuous age interactions")
    print(age_linear[age_linear.term.isin(POST_TERMS)].to_string(index=False))
    print("\nAge-specific spline estimates")
    print(age_spline[age_spline.term.isin(POST_TERMS)].to_string(index=False))
    print("\nSpline nonlinearity tests")
    print(age_nonlinearity[age_nonlinearity.term.isin(POST_TERMS)].to_string(index=False))
    print("\nExploratory IADL contextual interactions")
    print(context_interactions[context_interactions.term.isin(POST_TERMS)].to_string(index=False))
    print("\nContext variable audit")
    print(pd.DataFrame(context_audit).to_string(index=False))


if __name__ == "__main__":
    main()
