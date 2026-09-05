from __future__ import annotations

import warnings

from config import OUTPUT_DIR as OUT

import numpy as np
import pandas as pd
import pyhdfe
import statsmodels.api as sm
from scipy.stats import chi2, norm, t


import primary_analysis as locked
base = locked.base

IADL_ITEMS = ["money", "medications", "shopping", "meals"]
IADL_ROOTS = {
    "money": "moneya",
    "medications": "medsa",
    "shopping": "shopa",
    "meals": "mealsa",
}
EVENT_TERMS = locked.EVENT_TERMS
MODIFIERS = {
    "age_60_plus": "Age 60 years or older",
    "female": "Women versus men",
    "baseline_adl_difficulty": "Baseline ADL difficulty versus independence",
    "multimorbidity": "Two or more chronic conditions versus fewer than two",
}


def add_iadl(site: str, a: pd.DataFrame) -> pd.DataFrame:
    if site == "CHARLS":
        path = locked.CHARLS
        waves = locked.CHARLS_WAVES
        id_col = "ID"
    else:
        path = base.HRS
        waves = locked.HRS_WAVES
        id_col = "hhidpn"

    columns = [id_col]
    for wave in waves:
        columns.extend(f"r{wave}{root}" for root in IADL_ROOTS.values())
    raw = pd.read_stata(
        path,
        columns=columns,
        convert_categoricals=False,
        preserve_dtypes=False,
    )
    if site == "CHARLS":
        raw["ID"] = raw["ID"].astype(str)
    else:
        raw["ID"] = raw["hhidpn"].round().astype("Int64").astype(str)

    frames = []
    for wave in waves:
        z = raw[["ID", *[f"r{wave}{root}" for root in IADL_ROOTS.values()]]].copy()
        z = z.rename(columns={f"r{wave}{root}": item for item, root in IADL_ROOTS.items()})
        for item in IADL_ITEMS:
            z[item] = z[item].where(z[item].isin([0, 1]))
        z["wave"] = wave
        frames.append(z)
    iadl = pd.concat(frames, ignore_index=True)
    out = a.merge(iadl, on=["ID", "wave"], how="left", validate="one_to_one")
    out["iadl4"] = out[IADL_ITEMS].sum(axis=1, min_count=4)
    out["iadl4_any"] = np.where(
        out["iadl4"].notna(), out["iadl4"].gt(0).astype(float), np.nan
    )
    baseline_wave = waves[0]
    baseline = out.loc[out["wave"].eq(baseline_wave), ["ID", "iadl4"]].rename(
        columns={"iadl4": "base_iadl4"}
    )
    out = out.merge(baseline, on="ID", how="left", validate="many_to_one")
    out["age_60_plus"] = (
        out["r1agey"].ge(60) if site == "CHARLS" else out["r10agey_b"].ge(60)
    ).astype(float)
    out["baseline_adl_difficulty"] = np.where(
        out["base_adl5"].notna(), out["base_adl5"].gt(0).astype(float), np.nan
    )
    out["multimorbidity"] = np.where(
        out["comorbidity"].notna(), out["comorbidity"].ge(2).astype(float), np.nan
    )

    death_year = pd.to_numeric(
        out["death_year"] if site == "CHARLS" else out["radyear"], errors="coerce"
    ).where(lambda x: x.between(1900, 2030))
    out["died_before_wave"] = death_year.notna() & (
        death_year.lt(out["year"])
        | (death_year.eq(out["year"]) & ~out["inw"].eq(1))
    )
    out["death_or_adl_refined"] = np.where(
        out["died_before_wave"],
        1.0,
        np.where(out["inw"].eq(1) & out["adl5_any"].notna(), out["adl5_any"], np.nan),
    )
    return out


def baseline_sd(a: pd.DataFrame, wave: int, outcome: str) -> float:
    return float(a.loc[a["wave"].eq(wave), outcome].std())


def standardize(data: pd.DataFrame, sds: dict[str, float]) -> pd.DataFrame:
    out = data.copy()
    out["baseline_sd"] = out["cohort"].map(sds)
    for col in ["estimate", "std_error", "ci_low", "ci_high"]:
        out[f"standardized_{col}"] = out[col] / out["baseline_sd"]
    return out


def synthesize(data: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    differences = []
    summaries = []
    for term in EVENT_TERMS:
        z = data[data["term"].eq(term)].copy()
        if len(z) != 2:
            continue
        ch = z[z["cohort"].eq("CHARLS")].iloc[0]
        hr = z[z["cohort"].eq("HRS")].iloc[0]
        diff = hr.standardized_estimate - ch.standardized_estimate
        diff_se = np.sqrt(hr.standardized_std_error**2 + ch.standardized_std_error**2)
        differences.append({
            "outcome": outcome,
            "term": term,
            "HRS_minus_CHARLS_standardized": diff,
            "std_error": diff_se,
            "ci_low": diff - 1.96 * diff_se,
            "ci_high": diff + 1.96 * diff_se,
            "p_difference": 2 * norm.sf(abs(diff / diff_se)),
        })
        yi = z["standardized_estimate"].to_numpy(float)
        vi = z["standardized_std_error"].to_numpy(float) ** 2
        wi = 1 / vi
        fixed = np.sum(wi * yi) / np.sum(wi)
        fixed_se = np.sqrt(1 / np.sum(wi))
        q = np.sum(wi * (yi - fixed) ** 2)
        c = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0.0, (q - 1) / c) if c > 0 else 0.0
        wr = 1 / (vi + tau2)
        random = np.sum(wr * yi) / np.sum(wr)
        random_se = np.sqrt(1 / np.sum(wr))
        summaries.append({
            "outcome": outcome,
            "term": term,
            "k": 2,
            "fixed_effect_standardized_estimate": fixed,
            "fixed_std_error": fixed_se,
            "fixed_ci_low": fixed - 1.96 * fixed_se,
            "fixed_ci_high": fixed + 1.96 * fixed_se,
            "fixed_p_value": 2 * norm.sf(abs(fixed / fixed_se)),
            "random_effect_standardized_estimate": random,
            "random_std_error": random_se,
            "random_ci_low": random - 1.96 * random_se,
            "random_ci_high": random + 1.96 * random_se,
            "random_p_value": 2 * norm.sf(abs(random / random_se)),
            "tau_squared": tau2,
            "Q": q,
            "Q_p_value": chi2.sf(q, 1),
            "I_squared_pct": max(0.0, 100 * (q - 1) / q) if q > 0 else 0.0,
        })
    return pd.DataFrame(differences), pd.DataFrame(summaries)


def fit_interaction(
    stack: pd.DataFrame,
    outcome: str,
    modifier: str,
    label: str,
) -> list[dict]:
    z = stack[stack["inw"].eq(1) & stack[outcome].notna() & stack[modifier].notna()].copy()
    panel_n = z.groupby("person_fe").size()
    z = z[z["person_fe"].isin(panel_n[panel_n.ge(2)].index)].copy()
    interaction_terms = [f"{term}_x_modifier" for term in EVENT_TERMS]
    for term, interaction in zip(EVENT_TERMS, interaction_terms):
        z[interaction] = z[term] * z[modifier]
    design_terms = [*EVENT_TERMS, *interaction_terms]
    ids = np.column_stack([
        pd.factorize(z["person_fe"])[0],
        pd.factorize(z["wave_fe"])[0],
    ])
    algorithm = pyhdfe.create(ids, drop_singletons=False, residualize_method="map")
    y = algorithm.residualize(z[[outcome]].to_numpy(float)).ravel()
    x = algorithm.residualize(z[design_terms].to_numpy(float))
    keep = np.nanstd(x, axis=0) > 1e-12
    fit = sm.OLS(y, x[:, keep]).fit(cov_type="cluster", cov_kwds={"groups": z["ID"]})
    rows = []
    fitted_terms = np.asarray(design_terms)[keep]
    for term in EVENT_TERMS:
        name = f"{term}_x_modifier"
        if name not in fitted_terms:
            continue
        j = int(np.where(fitted_terms == name)[0][0])
        beta = float(fit.params[j])
        se = float(fit.bse[j])
        rows.append({
            "modifier": modifier,
            "modifier_label": label,
            "term": term,
            "n_rows": len(z),
            "n_unique_people": z["ID"].nunique(),
            "interaction_estimate": beta,
            "std_error": se,
            "ci_low": beta - 1.96 * se,
            "ci_high": beta + 1.96 * se,
            "p_interaction": float(fit.pvalues[j]),
        })
    return rows


def pooled_interactions(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (modifier, label, term), z in data.groupby(["modifier", "modifier_label", "term"]):
        if len(z) != 2:
            continue
        yi = z["standardized_interaction_estimate"].to_numpy(float)
        vi = z["standardized_std_error"].to_numpy(float) ** 2
        wi = 1 / vi
        beta = np.sum(wi * yi) / np.sum(wi)
        se = np.sqrt(1 / np.sum(wi))
        q = np.sum(wi * (yi - beta) ** 2)
        rows.append({
            "cohort": "Fixed-effect pooled",
            "modifier": modifier,
            "modifier_label": label,
            "term": term,
            "standardized_interaction_estimate": beta,
            "standardized_std_error": se,
            "standardized_ci_low": beta - 1.96 * se,
            "standardized_ci_high": beta + 1.96 * se,
            "p_interaction": 2 * norm.sf(abs(beta / se)),
            "Q": q,
            "Q_p_value": chi2.sf(q, 1),
        })
    return pd.DataFrame(rows)


def bh_by_scope(data: pd.DataFrame, scope_cols: list[str], p_col: str) -> pd.DataFrame:
    out = data.copy()
    out["q_value_bh"] = np.nan
    tested = out["term"].isin(["post1", "post2"])
    grouping = scope_cols[0] if len(scope_cols) == 1 else scope_cols
    for _, index in out[tested].groupby(grouping).groups.items():
        out.loc[index, "q_value_bh"] = base.bh_adjust(out.loc[index, p_col])
    return out


def missingness_table(site: str, a: pd.DataFrame, waves: list[int], years: dict[int, int]) -> pd.DataFrame:
    rows = []
    for wave in waves:
        interviewed = a[a["wave"].eq(wave) & a["inw"].eq(1)]
        for outcome, items in [("ADL5", locked.COMMON_ITEMS), ("IADL4", IADL_ITEMS)]:
            count = interviewed[items].notna().sum(axis=1)
            rows.append({
                "cohort": site,
                "survey_year": years[wave],
                "outcome": outcome,
                "interviewed_n": len(interviewed),
                "complete_item_set_n": int(count.eq(len(items)).sum()),
                "partial_item_missing_n": int(count.between(1, len(items) - 1).sum()),
                "all_items_missing_n": int(count.eq(0).sum()),
                "complete_item_set_pct": 100 * count.eq(len(items)).mean(),
            })
    return pd.DataFrame(rows)


def mice_imputations(
    site: str,
    a: pd.DataFrame,
    waves: list[int],
    event_cohorts: list[int],
    m: int = 20,
) -> pd.DataFrame:
    a = a.copy()
    a["_row_id"] = np.arange(len(a))
    interviewed = a[a["inw"].eq(1)].copy()
    age_col = "r1agey" if site == "CHARLS" else "r10agey_b"
    items = [*locked.COMMON_ITEMS, *IADL_ITEMS]
    base_covariates = interviewed[[age_col, "female", "comorbidity", "hospital"]].astype(float)
    for col in base_covariates:
        base_covariates[col] = base_covariates[col].fillna(base_covariates[col].median())
    wave_dummies = pd.get_dummies(interviewed["wave"].astype(int), prefix="wave", drop_first=True).astype(float)
    item_predictors = interviewed[items].astype(float).copy()
    for col in items:
        item_predictors[col] = item_predictors[col].fillna(item_predictors[col].mean())

    fitted_models = {}
    for item in items:
        predictors = pd.concat(
            [base_covariates, wave_dummies, item_predictors[[x for x in items if x != item]]],
            axis=1,
        )
        predictors = sm.add_constant(predictors, has_constant="add")
        observed = interviewed[item].notna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.GLM(
                interviewed.loc[observed, item].astype(float),
                predictors.loc[observed],
                family=sm.families.Binomial(),
            ).fit()
        covariance = np.asarray(fit.cov_params(), dtype=float)
        covariance = (covariance + covariance.T) / 2
        fitted_models[item] = (
            predictors.to_numpy(float),
            np.asarray(fit.params, dtype=float),
            covariance,
            interviewed[item].isna().to_numpy(),
        )

    stack, _ = base.make_stack(a, event_cohorts, "not_yet")
    model_rows = stack[stack["inw"].eq(1)].copy()
    panel_n = model_rows.groupby("person_fe").size()
    model_rows = model_rows[
        model_rows["person_fe"].isin(panel_n[panel_n.ge(2)].index)
    ].copy()
    fixed_effect_ids = np.column_stack([
        pd.factorize(model_rows["person_fe"])[0],
        pd.factorize(model_rows["wave_fe"])[0],
    ])
    absorber = pyhdfe.create(
        fixed_effect_ids, drop_singletons=False, residualize_method="map"
    )
    design = absorber.residualize(model_rows[EVENT_TERMS].to_numpy(float))
    keep = np.nanstd(design, axis=0) > 1e-12
    design = design[:, keep]
    fitted_terms = np.asarray(EVENT_TERMS)[keep]
    clusters = model_rows["ID"].to_numpy()
    row_ids = model_rows["_row_id"].astype(int).to_numpy()

    estimates = []
    rng = np.random.default_rng(20260902 if site == "CHARLS" else 20260903)
    for imputation in range(1, m + 1):
        completed = interviewed[items].astype(float).copy()
        for item in items:
            predictors, params, covariance, missing = fitted_models[item]
            try:
                draw = rng.multivariate_normal(params, covariance, check_valid="ignore")
            except np.linalg.LinAlgError:
                draw = params
            linear = np.clip(predictors @ draw, -20, 20)
            probability = 1 / (1 + np.exp(-linear))
            completed.loc[completed.index[missing], item] = rng.binomial(
                1, probability[missing]
            )
        z = a.copy()
        z.loc[interviewed.index, locked.COMMON_ITEMS] = completed[locked.COMMON_ITEMS].to_numpy()
        z.loc[interviewed.index, IADL_ITEMS] = completed[IADL_ITEMS].to_numpy()
        z["adl5_mi"] = z[locked.COMMON_ITEMS].sum(axis=1, min_count=5)
        z["iadl4_mi"] = z[IADL_ITEMS].sum(axis=1, min_count=4)
        outcomes = ["adl5_mi", "iadl4_mi"]
        y0 = np.column_stack([
            z[outcome].to_numpy(float)[row_ids] for outcome in outcomes
        ])
        y = absorber.residualize(y0)
        for outcome_index, outcome in enumerate(outcomes):
            fit = sm.OLS(y[:, outcome_index], design).fit(
                cov_type="cluster", cov_kwds={"groups": clusters}
            )
            for j, term in enumerate(fitted_terms):
                beta = float(fit.params[j])
                se = float(fit.bse[j])
                estimates.append({
                    "cohort": site,
                    "sensitivity": "multiple_imputation_items",
                    "outcome": outcome,
                    "term": term,
                    "n_rows": len(model_rows),
                    "n_unique_people": model_rows["ID"].nunique(),
                    "estimate": beta,
                    "std_error": se,
                    "ci_low": beta - 1.96 * se,
                    "ci_high": beta + 1.96 * se,
                    "p_value": float(fit.pvalues[j]),
                    "imputation": imputation,
                })
    estimates = pd.DataFrame(estimates)
    pooled = []
    for (cohort, outcome, term), z in estimates.groupby(["cohort", "outcome", "term"]):
        m_actual = len(z)
        qbar = z["estimate"].mean()
        ubar = np.mean(z["std_error"] ** 2)
        between = z["estimate"].var(ddof=1)
        total = ubar + (1 + 1 / m_actual) * between
        se = np.sqrt(total)
        if between > 0:
            df = (m_actual - 1) * (1 + ubar / ((1 + 1 / m_actual) * between)) ** 2
            p = 2 * t.sf(abs(qbar / se), df)
        else:
            df = np.inf
            p = 2 * norm.sf(abs(qbar / se))
        pooled.append({
            "cohort": cohort,
            "sensitivity": "multiple_imputation_items",
            "outcome": outcome.replace("_mi", ""),
            "term": term,
            "m": m_actual,
            "estimate": qbar,
            "std_error": se,
            "ci_low": qbar - 1.96 * se,
            "ci_high": qbar + 1.96 * se,
            "p_value": p,
            "rubin_df": df,
            "within_variance": ubar,
            "between_variance": between,
        })
    return pd.DataFrame(pooled)


def discrete_cif_matrix(status_matrix: np.ndarray) -> pd.DataFrame:
    active = np.ones(status_matrix.shape[0], dtype=bool)
    survival = 1.0
    cif_adl = 0.0
    cif_death = 0.0
    rows = []
    for relative_wave in range(status_matrix.shape[1]):
        current = status_matrix[:, relative_wave]
        known = active & np.isin(current, [0, 1, 2])
        n_risk = int(known.sum())
        d_adl = int((known & (current == 1)).sum())
        d_death = int((known & (current == 2)).sum())
        if n_risk:
            cif_adl += survival * d_adl / n_risk
            cif_death += survival * d_death / n_risk
            survival *= 1 - (d_adl + d_death) / n_risk
        active &= ~np.isin(current, [1, 2, 3])
        rows.append({
            "relative_wave": relative_wave,
            "n_at_risk": n_risk,
            "adl_events": d_adl,
            "death_events": d_death,
            "cif_adl": cif_adl,
            "cif_death": cif_death,
            "event_free_survival": survival,
        })
    return pd.DataFrame(rows)


def hrs_competing_outcomes(hrs: pd.DataFrame, rng_seed: int = 240902) -> pd.DataFrame:
    event_wave = 12
    index_rows = hrs[hrs["wave"].eq(event_wave)]
    eligible = set(
        index_rows.loc[index_rows["inw"].eq(1) & index_rows["adl5_any"].eq(0), "ID"]
    )
    cases = set(hrs.loc[hrs["incident_wave"].eq(event_wave), "ID"]) & eligible
    controls = set(hrs.loc[hrs["incident_wave"].isna(), "ID"]) & eligible
    matrices = {}
    for group, ids in [("First observed hospitalization", cases), ("Never hospitalized", controls)]:
        status_rows = []
        for person in sorted(ids):
            person_rows = hrs[hrs["ID"].eq(person)].set_index("wave")
            event_seen = False
            statuses = []
            for wave in [event_wave + 1, event_wave + 2]:
                if event_seen:
                    statuses.append(3)
                    continue
                row = person_rows.loc[wave]
                if bool(row["died_before_wave"]):
                    status = 2
                    event_seen = True
                elif row["inw"] == 1 and pd.notna(row["adl5_any"]):
                    status = 1 if row["adl5_any"] == 1 else 0
                    event_seen = status == 1
                else:
                    status = 3
                    event_seen = True
                statuses.append(status)
            status_rows.append(statuses)
        matrices[group] = np.asarray(status_rows, dtype=int)
    estimates = []
    for group, matrix in matrices.items():
        cif = discrete_cif_matrix(matrix)
        cif["relative_wave"] = cif["relative_wave"] + 1
        cif.insert(0, "group", group)
        cif["group_n"] = matrix.shape[0]
        estimates.append(cif)
    estimates = pd.concat(estimates, ignore_index=True)

    rng = np.random.default_rng(rng_seed)
    boot = []
    for b in range(500):
        sample_cif = {}
        for group, matrix in matrices.items():
            sampled = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
            sample_cif[group] = discrete_cif_matrix(matrix[sampled]).set_index(
                "relative_wave"
            )
        for matrix_wave, relative_wave in enumerate([1, 2]):
            boot.append({
                "bootstrap": b,
                "relative_wave": relative_wave,
                "adl_cif_difference": (
                    sample_cif["First observed hospitalization"].loc[matrix_wave, "cif_adl"]
                    - sample_cif["Never hospitalized"].loc[matrix_wave, "cif_adl"]
                ),
                "death_cif_difference": (
                    sample_cif["First observed hospitalization"].loc[matrix_wave, "cif_death"]
                    - sample_cif["Never hospitalized"].loc[matrix_wave, "cif_death"]
                ),
            })
    boot = pd.DataFrame(boot)
    comparison = []
    wide = estimates.pivot(index="relative_wave", columns="group", values=["cif_adl", "cif_death"])
    for relative_wave in [1, 2]:
        b = boot[boot["relative_wave"].eq(relative_wave)]
        comparison.append({
            "group": "Hospitalization minus never-hospitalized",
            "relative_wave": relative_wave,
            "group_n": len(cases),
            "comparison_group_n": len(controls),
            "cif_adl": (
                wide.loc[relative_wave, ("cif_adl", "First observed hospitalization")]
                - wide.loc[relative_wave, ("cif_adl", "Never hospitalized")]
            ),
            "cif_adl_ci_low": b["adl_cif_difference"].quantile(0.025),
            "cif_adl_ci_high": b["adl_cif_difference"].quantile(0.975),
            "cif_death": (
                wide.loc[relative_wave, ("cif_death", "First observed hospitalization")]
                - wide.loc[relative_wave, ("cif_death", "Never hospitalized")]
            ),
            "cif_death_ci_low": b["death_cif_difference"].quantile(0.025),
            "cif_death_ci_high": b["death_cif_difference"].quantile(0.975),
        })
    return pd.concat([estimates, pd.DataFrame(comparison)], ignore_index=True, sort=False)


def mortality_audit(charls: pd.DataFrame) -> pd.DataFrame:
    baseline = charls[charls["wave"].eq(1)].copy()
    death_year = pd.to_numeric(baseline["death_year"], errors="coerce").where(
        lambda x: x.between(1900, 2030)
    )
    harmonized = pd.read_stata(
        locked.CHARLS,
        columns=["ID", "radyear"],
        convert_categoricals=False,
        preserve_dtypes=False,
    )
    harmonized["ID"] = harmonized["ID"].astype(str)
    harmonized = harmonized[harmonized["ID"].isin(set(baseline["ID"]))]
    radyear = pd.to_numeric(harmonized["radyear"], errors="coerce").where(
        lambda x: x.between(1900, 2030)
    )
    return pd.DataFrame([
        {"metric": "Eligible baseline participants", "n": len(baseline), "note": "Analysis cohort"},
        {
            "metric": "Any death year identified",
            "n": int(death_year.notna().sum()),
            "note": "Combined Harmonized CHARLS and supplied 2020 exit module",
        },
        {
            "metric": "Death year in Harmonized CHARLS",
            "n": int(radyear.notna().sum()),
            "note": "Available harmonized death-year field",
        },
        {
            "metric": "Death identified before a modeled survey wave",
            "n": baseline["ID"].isin(charls.loc[charls["died_before_wave"], "ID"]).sum(),
            "note": "Lower-bound ascertainment; mortality follow-up remains incomplete",
        },
    ])


def add_iadl_baseline_rows(table1: pd.DataFrame, site: str, a: pd.DataFrame, baseline_wave: int, cohorts: list[int]) -> pd.DataFrame:
    row = locked.baseline_table(
        site,
        a,
        baseline_wave,
        cohorts,
        [("Common four-item IADL difficulty count", "iadl4", False)],
    )
    return pd.concat([table1, row], ignore_index=True)


def run():
    _, charls0 = locked.load_charls()
    _, hrs0 = base.load_hrs()
    charls = add_iadl("CHARLS", charls0)
    hrs = add_iadl("HRS", hrs0)

    stacks = {}
    iadl_rows = []
    iadl_item_rows = []
    interaction_rows = []
    mortality_rows = []
    for site, a, cohorts, baseline_wave, complete_n in [
        ("CHARLS", charls, locked.CHARLS_EVENT_COHORTS, 1, 4),
        ("HRS", hrs, locked.HRS_EVENT_COHORTS, 10, 5),
    ]:
        stack, _ = base.make_stack(a, cohorts, "not_yet")
        never_stack, _ = base.make_stack(a, cohorts, "never")
        stacks[site] = stack
        a["ipcw_iadl4"] = (
            locked.add_charls_ipcw(
                a.rename(columns={"base_adl5": "_base_adl5", "adl5": "_adl5"}).assign(
                    base_adl5=a["base_iadl4"], adl5=a["iadl4"]
                )
            )
            if site == "CHARLS"
            else base.add_ipcw(a, "iadl4")
        )
        stack = stack.merge(a[["ID", "wave", "ipcw_iadl4"]], on=["ID", "wave"], how="left")
        specifications = [
            ("main", stack, None, None),
            ("never_controls", never_stack, None, None),
            ("complete_panel", stack, None, complete_n),
            ("age_60_plus", stack[stack["age_60_plus"].eq(1)], None, None),
            ("survey_weighted", stack, "survey_weight", None),
            ("ipcw", stack, "ipcw_iadl4", None),
            ("baseline_independent", stack[stack["base_iadl4"].eq(0)], None, None),
        ]
        for label, z, weight_col, require_waves in specifications:
            rows = base.fit_fe(z, "iadl4", label, weight_col=weight_col, require_waves=require_waves)
            for row in rows:
                row["cohort"] = site
            iadl_rows.extend(rows)
        rows = base.fit_fe(stack, "iadl4_any", "binary_any_iadl")
        for row in rows:
            row["cohort"] = site
        iadl_rows.extend(rows)
        for item in IADL_ITEMS:
            rows = base.fit_fe(stack, item, "iadl_item")
            for row in rows:
                row["cohort"] = site
                row["item"] = item
            iadl_item_rows.extend(rows)
        for modifier, label in MODIFIERS.items():
            rows = fit_interaction(stack, "adl5", modifier, label)
            for row in rows:
                row["cohort"] = site
            interaction_rows.extend(rows)
        death_rows = base.fit_fe(
            stack,
            "death_or_adl_refined",
            "death_or_any_adl_refined" if site == "HRS" else "partially_ascertained_death_or_any_adl",
            include_deaths=True,
        )
        for row in death_rows:
            row["cohort"] = site
        mortality_rows.extend(death_rows)

    iadl = pd.DataFrame(iadl_rows)
    iadl_items = pd.DataFrame(iadl_item_rows)
    interactions = pd.DataFrame(interaction_rows)
    mortality = pd.DataFrame(mortality_rows)

    iadl_sds = {
        "CHARLS": baseline_sd(charls, 1, "iadl4"),
        "HRS": baseline_sd(hrs, 10, "iadl4"),
    }
    adl_sds = {
        "CHARLS": baseline_sd(charls, 1, "adl5"),
        "HRS": baseline_sd(hrs, 10, "adl5"),
    }
    iadl = standardize(iadl, iadl_sds)
    iadl_items["q_value_bh"] = np.nan
    tested = iadl_items["term"].isin(["post1", "post2"])
    for _, index in iadl_items[tested].groupby("cohort").groups.items():
        iadl_items.loc[index, "q_value_bh"] = base.bh_adjust(iadl_items.loc[index, "p_value"])

    interactions["baseline_sd"] = interactions["cohort"].map(adl_sds)
    for source, target in [
        ("interaction_estimate", "standardized_interaction_estimate"),
        ("std_error", "standardized_std_error"),
        ("ci_low", "standardized_ci_low"),
        ("ci_high", "standardized_ci_high"),
    ]:
        interactions[target] = interactions[source] / interactions["baseline_sd"]
    interactions = bh_by_scope(interactions, ["cohort"], "p_interaction")
    pooled_int = pooled_interactions(interactions)
    pooled_int = bh_by_scope(pooled_int, ["cohort"], "p_interaction")
    interaction_table = pd.concat([interactions, pooled_int], ignore_index=True, sort=False)

    iadl_main = iadl[iadl["sensitivity"].eq("main") & iadl["outcome"].eq("iadl4")].copy()
    iadl_differences, iadl_synthesis = synthesize(iadl_main, "iadl4")

    adl_main = pd.read_csv(OUT / "Table_2_primary_event_study.csv")
    adl_main["domain"] = "Basic ADL (5 items)"
    iadl_main["domain"] = "Instrumental ADL (4 items)"
    multidomain = pd.concat([adl_main, iadl_main], ignore_index=True, sort=False)

    existing_synthesis = pd.read_csv(OUT / "Table_8_two_cohort_synthesis.csv")
    existing_synthesis.insert(0, "outcome", "adl5")
    multidomain_synthesis = pd.concat(
        [existing_synthesis, iadl_synthesis], ignore_index=True, sort=False
    )

    missingness = pd.concat([
        missingness_table("CHARLS", charls, locked.CHARLS_WAVES, locked.CHARLS_YEARS),
        missingness_table("HRS", hrs, locked.HRS_WAVES, locked.HRS_YEARS),
    ], ignore_index=True)
    mi = pd.concat([
        mice_imputations("CHARLS", charls, locked.CHARLS_WAVES, locked.CHARLS_EVENT_COHORTS),
        mice_imputations("HRS", hrs, locked.HRS_WAVES, locked.HRS_EVENT_COHORTS),
    ], ignore_index=True)
    mi_sds = []
    for _, row in mi.iterrows():
        outcome_sds = adl_sds if row["outcome"] == "adl5" else iadl_sds
        row = row.copy()
        row["baseline_sd"] = outcome_sds[row["cohort"]]
        for col in ["estimate", "std_error", "ci_low", "ci_high"]:
            row[f"standardized_{col}"] = row[col] / row["baseline_sd"]
        mi_sds.append(row)
    mi = pd.DataFrame(mi_sds)

    old_sensitivity = pd.read_csv(OUT / "Table_3_sensitivity_analyses.csv")
    combined_sensitivity = pd.concat([old_sensitivity, iadl, mi], ignore_index=True, sort=False)

    table1 = pd.read_csv(OUT / "Table_1_baseline_characteristics.csv")
    table1 = add_iadl_baseline_rows(table1, "CHARLS", charls, 1, locked.CHARLS_EVENT_COHORTS)
    table1 = add_iadl_baseline_rows(table1, "HRS", hrs, 10, locked.HRS_EVENT_COHORTS)

    competing = hrs_competing_outcomes(hrs)
    charls_mortality = mortality_audit(charls)

    table1.to_csv(OUT / "Table_1_baseline_characteristics_upgraded.csv", index=False, encoding="utf-8-sig")
    multidomain.to_csv(OUT / "Table_2_multidomain_event_study.csv", index=False, encoding="utf-8-sig")
    combined_sensitivity.to_csv(OUT / "Table_3_sensitivity_analyses_upgraded.csv", index=False, encoding="utf-8-sig")
    interaction_table.to_csv(OUT / "Table_4_effect_modification_interactions.csv", index=False, encoding="utf-8-sig")
    iadl_items.to_csv(OUT / "Table_S2_IADL_item_analyses.csv", index=False, encoding="utf-8-sig")
    missingness.to_csv(OUT / "Table_S3_missing_data_audit.csv", index=False, encoding="utf-8-sig")
    mi.to_csv(OUT / "Table_S4_multiple_imputation_results.csv", index=False, encoding="utf-8-sig")
    competing.to_csv(OUT / "Table_S5_HRS_competing_outcomes.csv", index=False, encoding="utf-8-sig")
    charls_mortality.to_csv(OUT / "Table_S6_CHARLS_mortality_audit.csv", index=False, encoding="utf-8-sig")
    mortality.to_csv(OUT / "Table_S7_death_composite_analyses.csv", index=False, encoding="utf-8-sig")
    iadl_differences.to_csv(OUT / "Table_7_IADL_cross_cohort_differences.csv", index=False, encoding="utf-8-sig")
    multidomain_synthesis.to_csv(OUT / "Table_9_multidomain_two_cohort_synthesis.csv", index=False, encoding="utf-8-sig")

    multidomain.to_csv(OUT / "Source_Data_Figure_1_upgraded.csv", index=False, encoding="utf-8-sig")
    combined_sensitivity[
        combined_sensitivity["term"].isin(["post1", "post2"])
    ].to_csv(OUT / "Source_Data_Figure_2_upgraded.csv", index=False, encoding="utf-8-sig")
    interaction_table[
        interaction_table["term"].isin(["post1", "post2"])
    ].to_csv(OUT / "Source_Data_Figure_S2_upgraded.csv", index=False, encoding="utf-8-sig")
    competing.to_csv(OUT / "Source_Data_Figure_S4.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for domain, data in [("ADL5", adl_main), ("IADL4", iadl_main)]:
        for _, row in data[data["term"].isin(["post1", "post2"])].iterrows():
            summary_rows.append({
                "domain": domain,
                "cohort": row["cohort"],
                "term": row["term"],
                "estimate": row["estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "standardized_estimate": row["standardized_estimate"],
                "standardized_ci_low": row["standardized_ci_low"],
                "standardized_ci_high": row["standardized_ci_high"],
                "p_value": row["p_value"],
            })
    pd.DataFrame(summary_rows).to_csv(
        OUT / "upgraded_key_results.csv", index=False, encoding="utf-8-sig"
    )

    print("IADL primary results")
    print(iadl_main.to_string(index=False))
    print("\nFormal effect-modification tests (post-event only)")
    print(interaction_table[interaction_table["term"].isin(["post1", "post2"])].to_string(index=False))
    print("\nMultiple-imputation sensitivity")
    print(mi[mi["term"].isin(["post1", "post2"])].to_string(index=False))
    print("\nHRS competing-outcome cumulative incidence")
    print(competing.to_string(index=False))


if __name__ == "__main__":
    run()
