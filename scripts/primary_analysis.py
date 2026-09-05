from config import CHARLS_DATA as CHARLS, CHARLS_EXIT_DATA as CHARLS_EXIT, OUTPUT_DIR as OUT

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, norm

import hrs_core as base
CHARLS_WAVES = [1, 2, 3, 4]
CHARLS_YEARS = {1: 2011, 2: 2013, 3: 2015, 4: 2018}
CHARLS_EVENT_COHORTS = [2, 3]
HRS_WAVES = [10, 11, 12, 13, 14]
HRS_YEARS = {10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018}
HRS_EVENT_COHORTS = [11, 12, 13]
COMMON_ITEMS = ["dress", "bathe", "eat", "bed", "toilet"]
EVENT_TERMS = ["pre2", "index", "post1", "post2"]


def load_charls():
    item_roots = {
        "dress": "dressa", "bathe": "batha", "eat": "eata",
        "bed": "beda", "toilet": "toilta",
    }
    chronic_roots = [
        "hibpe", "diabe", "dyslipe", "hearte", "stroke", "cancre",
        "lunge", "livere", "kidneye", "digeste", "arthre", "asthmae", "memrye",
    ]
    baseline = [
        "ID", "r1agey", "ragender", "raeducl", "r1mstat", "h1rural",
        "r1mbmi", "r1smokev", "r1smoken", "r1drinkl", "r1higov",
        "r1wtrespb", "radyear", "inw1", "inw2", "inw3", "inw4",
    ]
    wave_cols = []
    for wave in CHARLS_WAVES:
        wave_cols += [
            f"r{wave}hosp1y", f"r{wave}hearte", f"r{wave}stroke", f"r{wave}cancre",
            *[f"r{wave}{root}" for root in item_roots.values()],
        ]
    baseline += [f"r1{root}" for root in chronic_roots]
    h = pd.read_stata(
        CHARLS, columns=list(dict.fromkeys(baseline + wave_cols)),
        convert_categoricals=False, preserve_dtypes=False,
    )
    h["ID"] = h["ID"].astype(str)
    h = h[h["inw1"].eq(1) & h["r1agey"].ge(50) & h["r1hosp1y"].eq(0)].copy()

    exit_data = pd.read_stata(
        CHARLS_EXIT, columns=["ID", "exb001_1"],
        convert_categoricals=False, preserve_dtypes=False,
    ).rename(columns={"exb001_1": "exit_death_year"})
    exit_data["ID"] = exit_data["ID"].astype(str)
    exit_data = exit_data.drop_duplicates("ID")
    h = h.merge(exit_data, on="ID", how="left", validate="one_to_one")
    h["death_year"] = h["exit_death_year"].combine_first(h["radyear"])

    h["female"] = h["ragender"].eq(2).astype(float)
    h["high_school_plus"] = h["raeducl"].ge(2).astype(float)
    h["married_partnered"] = h["r1mstat"].isin([1, 3]).astype(float)
    h["rural"] = h["h1rural"].where(h["h1rural"].isin([0, 1]))
    h["bmi"] = h["r1mbmi"].where(h["r1mbmi"].between(10, 60))
    h["ever_smoke"] = h["r1smokev"].where(h["r1smokev"].isin([0, 1]))
    h["current_smoke"] = h["r1smoken"].where(h["r1smoken"].isin([0, 1]))
    h["current_drink"] = h["r1drinkl"].where(h["r1drinkl"].isin([0, 1]))
    h["insurance"] = h["r1higov"].where(h["r1higov"].isin([0, 1]))
    for root in chronic_roots:
        h[f"r1{root}"] = h[f"r1{root}"].where(h[f"r1{root}"].isin([0, 1]))
    h["comorbidity"] = h[[f"r1{x}" for x in chronic_roots]].sum(axis=1, min_count=8)
    h["survey_weight"] = h["r1wtrespb"]
    positive = h.loc[h["survey_weight"].gt(0), "survey_weight"]
    if len(positive):
        lo, hi = positive.quantile([0.01, 0.99])
        h["survey_weight"] = h["survey_weight"].clip(lo, hi)

    keep_base = [
        "ID", "r1agey", "female", "high_school_plus", "married_partnered",
        "rural", "bmi", "ever_smoke", "current_smoke", "current_drink",
        "insurance", "comorbidity", "survey_weight", "death_year",
    ]
    frames = []
    for wave in CHARLS_WAVES:
        cols = [
            "ID", f"inw{wave}", f"r{wave}hosp1y", f"r{wave}hearte",
            f"r{wave}stroke", f"r{wave}cancre",
            *[f"r{wave}{root}" for root in item_roots.values()],
        ]
        x = h[cols].copy()
        rename = {
            f"inw{wave}": "inw", f"r{wave}hosp1y": "hospital",
            f"r{wave}hearte": "heart", f"r{wave}stroke": "stroke",
            f"r{wave}cancre": "cancer",
        }
        rename.update({f"r{wave}{root}": item for item, root in item_roots.items()})
        x = x.rename(columns=rename)
        for col in ["hospital", *COMMON_ITEMS, "heart", "stroke", "cancer"]:
            x[col] = x[col].where(x[col].isin([0, 1]))
        x["adl5"] = x[COMMON_ITEMS].sum(axis=1, min_count=5)
        x["adl5_any"] = np.where(x["adl5"].notna(), x["adl5"].gt(0).astype(float), np.nan)
        x["wave"] = wave
        x["year"] = CHARLS_YEARS[wave]
        x = x.merge(h[keep_base], on="ID", how="left", validate="many_to_one")
        frames.append(x)
    a = pd.concat(frames, ignore_index=True).sort_values(["ID", "wave"]).reset_index(drop=True)

    for disease in ["heart", "stroke", "cancer"]:
        previous = a.groupby("ID")[disease].shift(1)
        a[f"new_{disease}"] = a[disease].eq(1) & previous.eq(0)
    a["new_major"] = a[["new_heart", "new_stroke", "new_cancer"]].any(axis=1)
    incident = (
        a[a["wave"].gt(1) & a["hospital"].eq(1)]
        .sort_values(["ID", "wave"]).drop_duplicates("ID")[["ID", "wave"]]
        .rename(columns={"wave": "incident_wave"})
    )
    a = a.merge(incident, on="ID", how="left")
    major_ids = set(a.loc[
        a["incident_wave"].notna() & a["wave"].eq(a["incident_wave"]) & a["new_major"], "ID"
    ])
    a["major_at_hospitalization"] = a["ID"].isin(major_ids)
    a["died_by_wave"] = a["death_year"].notna() & a["death_year"].le(a["year"])
    a["death_or_adl"] = np.where(
        a["died_by_wave"], 1.0,
        np.where(a["inw"].eq(1) & a["adl5_any"].notna(), a["adl5_any"], np.nan),
    )
    baseline_adl = a[a["wave"].eq(1)][["ID", "adl5"]].rename(columns={"adl5": "base_adl5"})
    a = a.merge(baseline_adl, on="ID", how="left", validate="many_to_one")
    return h, a


def add_charls_ipcw(a):
    result = pd.Series(np.nan, index=a.index, dtype=float)
    predictors = [
        "r1agey", "female", "high_school_plus", "married_partnered", "rural",
        "bmi", "ever_smoke", "current_smoke", "current_drink", "insurance",
        "comorbidity", "base_adl5",
    ]
    for wave in CHARLS_WAVES:
        idx = a["wave"].eq(wave)
        z = a.loc[idx, predictors].copy()
        observed = (a.loc[idx, "inw"].eq(1) & a.loc[idx, "adl5"].notna()).astype(float)
        if wave == 1:
            result.loc[idx & a["adl5"].notna()] = 1.0
            continue
        for col in z.columns:
            z[col] = z[col].fillna(z[col].median())
        X = sm.add_constant(z.astype(float), has_constant="add")
        fit = sm.GLM(observed.to_numpy(), X.to_numpy(), family=sm.families.Binomial()).fit()
        probability = np.clip(fit.predict(X.to_numpy()), 0.03, 0.995)
        weights = observed.mean() / probability
        observed_weights = weights[observed.to_numpy().astype(bool)]
        lo, hi = np.quantile(observed_weights, [0.01, 0.99])
        weights = np.clip(weights, lo, hi)
        result.loc[idx] = np.where(observed.to_numpy().astype(bool), weights, np.nan)
    return result


def run_site_models(
    site, a, waves, years, event_cohorts, age_col, complete_n, middle_cohort,
    include_death_composite,
):
    main_stack, main_counts = base.make_stack(a, event_cohorts, "not_yet")
    never_stack, never_counts = base.make_stack(a, event_cohorts, "never")
    result_rows = []
    specifications = [
        ("main", main_stack, None, None, False),
        ("never_controls", never_stack, None, None, False),
        ("complete_panel", main_stack, None, complete_n, False),
        ("age_60_plus", main_stack[main_stack[age_col].ge(60)], None, None, False),
        ("exclude_major_disease_onset", main_stack[~(
            main_stack["case"].eq(1) & main_stack["major_at_hospitalization"]
        )], None, None, False),
        ("survey_weighted", main_stack, "survey_weight", None, False),
        ("ipcw", main_stack, "ipcw_adl5", None, False),
        ("baseline_independent", main_stack[main_stack["base_adl5"].eq(0)], None, None, False),
        ("middle_event_cohort", main_stack[main_stack["cohort"].eq(middle_cohort)], None, None, False),
    ]
    for label, stack, weight_col, require_waves, include_deaths in specifications:
        result_rows += base.fit_fe(
            stack, "adl5", label, weight_col=weight_col,
            require_waves=require_waves, include_deaths=include_deaths,
        )
    result_rows += base.fit_fe(main_stack, "adl5_any", "binary_any_adl")
    if include_death_composite:
        result_rows += base.fit_fe(
            main_stack, "death_or_adl", "death_or_any_adl", include_deaths=True
        )
    results = pd.DataFrame(result_rows)
    results.insert(0, "cohort", site)

    item_rows = []
    for item in COMMON_ITEMS:
        rows = base.fit_fe(main_stack, item, "adl_item")
        for row in rows:
            row["item"] = item
        item_rows += rows
    items = pd.DataFrame(item_rows)
    items.insert(0, "cohort", site)
    items["q_value_bh"] = np.nan
    tested = items["term"].isin(["post1", "post2"])
    items.loc[tested, "q_value_bh"] = base.bh_adjust(items.loc[tested, "p_value"])

    subgroup_rows = []
    for label, stack in [
        ("Age 50-59", main_stack[main_stack[age_col].lt(60)]),
        ("Age 60+", main_stack[main_stack[age_col].ge(60)]),
        ("Men", main_stack[main_stack["female"].eq(0)]),
        ("Women", main_stack[main_stack["female"].eq(1)]),
        ("Baseline ADL independent", main_stack[main_stack["base_adl5"].eq(0)]),
        ("Baseline ADL difficulty", main_stack[main_stack["base_adl5"].gt(0)]),
    ]:
        subgroup_rows += base.fit_fe(stack, "adl5", label)
    subgroups = pd.DataFrame(subgroup_rows)
    subgroups.insert(0, "cohort", site)

    cohort_rows = []
    for cohort_wave in event_cohorts:
        rows = base.fit_fe(
            main_stack[main_stack["cohort"].eq(cohort_wave)], "adl5",
            f"{years[cohort_wave]} incident cohort",
        )
        for row in rows:
            row["event_cohort_year"] = years[cohort_wave]
        cohort_rows += rows
    cohorts = pd.DataFrame(cohort_rows)
    cohorts.insert(0, "cohort", site)

    all_counts = pd.concat([main_counts, never_counts], ignore_index=True)
    all_counts.insert(0, "cohort", site)
    all_counts["cohort_year"] = all_counts["cohort_wave"].map(years)
    return results, items, subgroups, cohorts, all_counts, main_stack


def standardized_difference(case, control, binary=False):
    x, y = case.dropna().astype(float), control.dropna().astype(float)
    if not len(x) or not len(y):
        return np.nan
    if binary:
        p1, p0 = x.mean(), y.mean()
        denominator = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
    else:
        denominator = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return (x.mean() - y.mean()) / denominator if denominator > 0 else 0.0


def format_value(series, binary=False):
    x = series.dropna().astype(float)
    if binary:
        return f"{int(x.eq(1).sum())} ({100 * x.mean():.1f}%)"
    return f"{x.mean():.2f} ({x.std():.2f})"


def baseline_table(site, a, baseline_wave, event_cohorts, variables):
    b = a[a["wave"].eq(baseline_wave)].copy()
    case = b[b["incident_wave"].isin(event_cohorts)]
    control = b[~b["incident_wave"].isin(event_cohorts)]
    rows = []
    for label, column, binary in variables:
        rows.append({
            "cohort": site,
            "characteristic": label,
            "incident_hospitalization": format_value(case[column], binary),
            "at_risk_comparison": format_value(control[column], binary),
            "absolute_standardized_difference": abs(
                standardized_difference(case[column], control[column], binary)
            ),
        })
    return pd.DataFrame(rows)


def flow_and_attrition(
    site, a, baseline_wave, waves, years, event_cohorts, death_ascertainment_complete,
):
    baseline = a[a["wave"].eq(baseline_wave)]
    flow = [{
        "cohort": site,
        "stage": "Eligible baseline respondents aged 50+ without interval hospitalization",
        "n": baseline["ID"].nunique(),
    }]
    for wave in waves[1:]:
        flow.append({
            "cohort": site, "stage": f"First-observed hospitalization in {years[wave]}",
            "n": int(baseline["incident_wave"].eq(wave).sum()),
        })
    flow.append({
        "cohort": site, "stage": f"No hospitalization observed through {years[waves[-1]]}",
        "n": int(baseline["incident_wave"].isna().sum()),
    })
    attrition = []
    for event_wave in event_cohorts:
        ids = set(baseline.loc[baseline["incident_wave"].eq(event_wave), "ID"])
        for survey_wave in waves:
            if survey_wave < event_wave or survey_wave > event_wave + 2:
                continue
            z = a[a["ID"].isin(ids) & a["wave"].eq(survey_wave)]
            attrition.append({
                "cohort": site, "event_cohort_year": years[event_wave],
                "relative_wave": survey_wave - event_wave,
                "survey_year": years[survey_wave], "cohort_n": len(ids),
                "adl5_observed_n": int(z["adl5"].notna().sum()),
                "died_by_wave_n": (
                    int(z["died_by_wave"].sum()) if death_ascertainment_complete else np.nan
                ),
                "death_ascertainment": (
                    "available" if death_ascertainment_complete
                    else "incomplete; not used for cross-cohort inference"
                ),
            })
    return pd.DataFrame(flow), pd.DataFrame(attrition)


def synthesize(main_results, baseline_sd):
    main = main_results[
        main_results["sensitivity"].eq("main") & main_results["outcome"].eq("adl5")
    ].copy()
    main["baseline_sd"] = main["cohort"].map(baseline_sd)
    for column in ["estimate", "std_error", "ci_low", "ci_high"]:
        main[f"standardized_{column}"] = main[column] / main["baseline_sd"]

    comparison_rows = []
    synthesis_rows = []
    for term in EVENT_TERMS:
        z = main[main["term"].eq(term)].copy()
        if len(z) != 2:
            continue
        china = z[z["cohort"].eq("CHARLS")].iloc[0]
        usa = z[z["cohort"].eq("HRS")].iloc[0]
        difference = usa["standardized_estimate"] - china["standardized_estimate"]
        difference_se = np.sqrt(
            usa["standardized_std_error"] ** 2 + china["standardized_std_error"] ** 2
        )
        comparison_rows.append({
            "term": term, "HRS_minus_CHARLS_standardized": difference,
            "std_error": difference_se,
            "ci_low": difference - 1.96 * difference_se,
            "ci_high": difference + 1.96 * difference_se,
            "p_difference": 2 * norm.sf(abs(difference / difference_se)),
        })

        yi = z["standardized_estimate"].to_numpy(float)
        vi = z["standardized_std_error"].to_numpy(float) ** 2
        wi = 1 / vi
        fixed = np.sum(wi * yi) / np.sum(wi)
        fixed_se = np.sqrt(1 / np.sum(wi))
        q = np.sum(wi * (yi - fixed) ** 2)
        c = np.sum(wi) - np.sum(wi ** 2) / np.sum(wi)
        tau2 = max(0.0, (q - 1) / c) if c > 0 else 0.0
        wr = 1 / (vi + tau2)
        random = np.sum(wr * yi) / np.sum(wr)
        random_se = np.sqrt(1 / np.sum(wr))
        synthesis_rows.append({
            "term": term, "k": 2,
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
            "tau_squared": tau2, "Q": q, "Q_p_value": chi2.sf(q, 1),
            "I_squared_pct": max(0.0, 100 * (q - 1) / q) if q > 0 else 0.0,
        })
    return main, pd.DataFrame(comparison_rows), pd.DataFrame(synthesis_rows)


def add_standardized_columns(data, baseline_sd):
    out = data.copy()
    out["baseline_sd"] = out["cohort"].map(baseline_sd)
    mask = out["outcome"].eq("adl5")
    for column in ["estimate", "std_error", "ci_low", "ci_high"]:
        out[f"standardized_{column}"] = np.where(
            mask, out[column] / out["baseline_sd"], np.nan
        )
    return out


def p_text(value):
    if pd.isna(value):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def make_formatted_primary(main):
    labels = {
        "pre2": "2+ waves before hospitalization",
        "index": "Hospitalization wave",
        "post1": "1 wave after hospitalization",
        "post2": "2+ waves after hospitalization",
    }
    out = main.copy()
    out["relative_time"] = out["term"].map(labels)
    out["ADL_count_difference_95CI"] = out.apply(
        lambda row: f"{row.estimate:.3f} ({row.ci_low:.3f} to {row.ci_high:.3f})", axis=1
    )
    out["standardized_difference_95CI"] = out.apply(
        lambda row: (
            f"{row.standardized_estimate:.3f} "
            f"({row.standardized_ci_low:.3f} to {row.standardized_ci_high:.3f})"
        ), axis=1
    )
    out["P_value"] = out["p_value"].map(p_text)
    return out[[
        "cohort", "relative_time", "n_unique_people", "ADL_count_difference_95CI",
        "standardized_difference_95CI", "P_value",
    ]]


def make_formatted_sensitivity(results):
    labels = {
        "main": "Primary model",
        "never_controls": "Never-hospitalized controls",
        "complete_panel": "Complete panel",
        "age_60_plus": "Age 60+",
        "exclude_major_disease_onset": "Exclude major disease onset",
        "survey_weighted": "Survey weighted",
        "ipcw": "Attrition weighted",
        "baseline_independent": "Baseline ADL independent",
        "middle_event_cohort": "Middle event cohort",
    }
    z = results[
        results["outcome"].eq("adl5") & results["sensitivity"].isin(labels)
    ].copy()
    rows = []
    for (cohort, sensitivity), group in z.groupby(["cohort", "sensitivity"], sort=False):
        indexed = group.set_index("term")
        row = {"cohort": cohort, "analysis": labels[sensitivity]}
        for term, prefix in [("pre2", "pretrend"), ("post1", "one_wave_after"), ("post2", "two_plus_waves_after")]:
            if term not in indexed.index:
                row[f"{prefix}_estimate_95CI"] = ""
                row[f"{prefix}_P_value"] = ""
                continue
            value = indexed.loc[term]
            row[f"{prefix}_estimate_95CI"] = (
                f"{value.estimate:.3f} ({value.ci_low:.3f} to {value.ci_high:.3f})"
            )
            row[f"{prefix}_P_value"] = p_text(value.p_value)
        rows.append(row)
    return pd.DataFrame(rows)


def run():
    _, charls = load_charls()
    _, hrs = base.load_hrs()
    charls["ipcw_adl5"] = add_charls_ipcw(charls)
    hrs["ipcw_adl5"] = base.add_ipcw(hrs, "adl5")

    charls_outputs = run_site_models(
        "CHARLS", charls, CHARLS_WAVES, CHARLS_YEARS, CHARLS_EVENT_COHORTS,
        "r1agey", 4, 3, False,
    )
    hrs_outputs = run_site_models(
        "HRS", hrs, HRS_WAVES, HRS_YEARS, HRS_EVENT_COHORTS,
        "r10agey_b", 5, 12, True,
    )
    results = pd.concat([charls_outputs[0], hrs_outputs[0]], ignore_index=True)
    items = pd.concat([charls_outputs[1], hrs_outputs[1]], ignore_index=True)
    subgroups = pd.concat([charls_outputs[2], hrs_outputs[2]], ignore_index=True)
    event_cohorts = pd.concat([charls_outputs[3], hrs_outputs[3]], ignore_index=True)
    cohort_counts = pd.concat([charls_outputs[4], hrs_outputs[4]], ignore_index=True)

    baseline_sd = {
        "CHARLS": charls.loc[charls["wave"].eq(1), "adl5"].std(),
        "HRS": hrs.loc[hrs["wave"].eq(10), "adl5"].std(),
    }
    results = add_standardized_columns(results, baseline_sd)
    subgroups = add_standardized_columns(subgroups, baseline_sd)
    event_cohorts = add_standardized_columns(event_cohorts, baseline_sd)

    charls_vars = [
        ("Age, years", "r1agey", False), ("Female", "female", True),
        ("High school/vocational education or above", "high_school_plus", True),
        ("Married/partnered", "married_partnered", True), ("Rural residence", "rural", True),
        ("BMI, kg/m2", "bmi", False), ("Ever smoked", "ever_smoke", True),
        ("Current smoker", "current_smoke", True), ("Current drinking", "current_drink", True),
        ("Public health insurance", "insurance", True),
        ("Chronic condition count", "comorbidity", False),
        ("Common five-item ADL difficulty count", "adl5", False),
    ]
    hrs_vars = [
        ("Age, years", "r10agey_b", False), ("Female", "female", True),
        ("Hispanic", "hispanic", True), ("Black, non-Hispanic", "black_nonhispanic", True),
        ("High school graduate or above", "high_school_plus", True),
        ("Married/partnered", "married_partnered", True), ("BMI, kg/m2", "bmi", False),
        ("Ever smoked", "ever_smoke", True), ("Current smoker", "current_smoke", True),
        ("Current drinking", "current_drink", True),
        ("Chronic condition count", "comorbidity", False),
        ("Common five-item ADL difficulty count", "adl5", False),
    ]
    table1 = pd.concat([
        baseline_table("CHARLS", charls, 1, CHARLS_EVENT_COHORTS, charls_vars),
        baseline_table("HRS", hrs, 10, HRS_EVENT_COHORTS, hrs_vars),
    ], ignore_index=True)

    charls_flow, charls_attrition = flow_and_attrition(
        "CHARLS", charls, 1, CHARLS_WAVES, CHARLS_YEARS, CHARLS_EVENT_COHORTS, False
    )
    hrs_flow, hrs_attrition = flow_and_attrition(
        "HRS", hrs, 10, HRS_WAVES, HRS_YEARS, HRS_EVENT_COHORTS, True
    )
    flow = pd.concat([charls_flow, hrs_flow], ignore_index=True)
    attrition = pd.concat([charls_attrition, hrs_attrition], ignore_index=True)

    main, differences, synthesis = synthesize(results, baseline_sd)
    summary = pd.DataFrame([
        {
            "cohort": "CHARLS", "baseline_year": 2011,
            "eligible_baseline_n": charls.loc[charls["wave"].eq(1), "ID"].nunique(),
            "primary_event_cases_n": int(charls.loc[
                charls["wave"].eq(1), "incident_wave"
            ].isin(CHARLS_EVENT_COHORTS).sum()),
            "baseline_adl5_mean": charls.loc[charls["wave"].eq(1), "adl5"].mean(),
            "baseline_adl5_sd": baseline_sd["CHARLS"],
        },
        {
            "cohort": "HRS", "baseline_year": 2010,
            "eligible_baseline_n": hrs.loc[hrs["wave"].eq(10), "ID"].nunique(),
            "primary_event_cases_n": int(hrs.loc[
                hrs["wave"].eq(10), "incident_wave"
            ].isin(HRS_EVENT_COHORTS).sum()),
            "baseline_adl5_mean": hrs.loc[hrs["wave"].eq(10), "adl5"].mean(),
            "baseline_adl5_sd": baseline_sd["HRS"],
        },
    ])

    table1.to_csv(OUT / "Table_1_baseline_characteristics.csv", index=False, encoding="utf-8-sig")
    main.to_csv(OUT / "Table_2_primary_event_study.csv", index=False, encoding="utf-8-sig")
    make_formatted_primary(main).to_csv(
        OUT / "Table_2_primary_event_study_formatted.csv", index=False, encoding="utf-8-sig"
    )
    results.to_csv(OUT / "Table_3_sensitivity_analyses.csv", index=False, encoding="utf-8-sig")
    make_formatted_sensitivity(results).to_csv(
        OUT / "Table_3_sensitivity_analyses_formatted.csv", index=False, encoding="utf-8-sig"
    )
    subgroups.to_csv(OUT / "Table_4_subgroup_analyses.csv", index=False, encoding="utf-8-sig")
    items.to_csv(OUT / "Table_5_ADL_item_analyses.csv", index=False, encoding="utf-8-sig")
    event_cohorts.to_csv(OUT / "Table_6_event_cohort_analyses.csv", index=False, encoding="utf-8-sig")
    differences.to_csv(OUT / "Table_7_cross_cohort_differences.csv", index=False, encoding="utf-8-sig")
    synthesis.to_csv(OUT / "Table_8_two_cohort_synthesis.csv", index=False, encoding="utf-8-sig")
    attrition.to_csv(OUT / "Table_S1_attrition_and_death.csv", index=False, encoding="utf-8-sig")
    flow.to_csv(OUT / "cohort_flow.csv", index=False, encoding="utf-8-sig")
    cohort_counts.to_csv(OUT / "event_cohort_counts.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "study_summary.csv", index=False, encoding="utf-8-sig")

    figure1 = main.copy()
    references = pd.DataFrame([
        {**{column: np.nan for column in figure1.columns}, "cohort": site, "term": "reference",
         "estimate": 0.0, "std_error": 0.0, "ci_low": 0.0, "ci_high": 0.0,
         "standardized_estimate": 0.0, "standardized_std_error": 0.0,
         "standardized_ci_low": 0.0, "standardized_ci_high": 0.0}
        for site in ["CHARLS", "HRS"]
    ])
    pooled = synthesis.rename(columns={
        "fixed_effect_standardized_estimate": "standardized_estimate",
        "fixed_std_error": "standardized_std_error",
        "fixed_ci_low": "standardized_ci_low",
        "fixed_ci_high": "standardized_ci_high",
        "fixed_p_value": "p_value",
    })[[
        "term", "standardized_estimate", "standardized_std_error",
        "standardized_ci_low", "standardized_ci_high", "p_value",
    ]].copy()
    pooled["cohort"] = "Fixed-effect pooled"
    pooled["sensitivity"] = "fixed_effect_synthesis"
    pooled["outcome"] = "adl5_standardized"
    pd.concat([figure1, references, pooled], ignore_index=True).to_csv(
        OUT / "Source_Data_Figure_1.csv", index=False, encoding="utf-8-sig"
    )
    results[
        results["outcome"].eq("adl5") & results["sensitivity"].isin([
            "main", "never_controls", "complete_panel", "age_60_plus",
            "exclude_major_disease_onset", "survey_weighted", "ipcw",
            "baseline_independent", "middle_event_cohort",
        ])
    ].to_csv(OUT / "Source_Data_Figure_2.csv", index=False, encoding="utf-8-sig")
    items.to_csv(OUT / "Source_Data_Figure_S1.csv", index=False, encoding="utf-8-sig")
    subgroups.to_csv(OUT / "Source_Data_Figure_S2.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print("\nPrimary harmonized five-item ADL results")
    print(main.to_string(index=False))
    print("\nStandardized cross-cohort differences")
    print(differences.to_string(index=False))
    print("\nTwo-cohort standardized synthesis")
    print(synthesis.to_string(index=False))


if __name__ == "__main__":
    run()
