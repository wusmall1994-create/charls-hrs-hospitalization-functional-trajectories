from config import CHARLS_DATA as CHARLS, HRS_DATA as HRS, OUTPUT_DIR as OUT

import numpy as np
import pandas as pd
import pyhdfe
import statsmodels.api as sm
from scipy.stats import norm, chi2

HRS_WAVES = [10, 11, 12, 13, 14]
HRS_YEARS = {10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018}
HRS_EVENT_COHORTS = [11, 12, 13]
CHARLS_WAVES = [1, 2, 3, 4]
CHARLS_YEARS = {1: 2011, 2: 2013, 3: 2015, 4: 2018}
CHARLS_EVENT_COHORTS = [2, 3]
EVENT_TERMS = ["pre2", "index", "post1", "post2"]
COMMON_ITEMS = ["dress", "bathe", "eat", "bed", "toilet"]


def bh_adjust(values):
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def load_hrs():
    baseline = [
        "hhidpn", "ragender", "rahispan", "raracem", "raeduc", "radyear",
        "r10agey_b", "r10mstat", "r10bmi", "r10smokev", "r10smoken",
        "r10drink", "r10hibpe", "r10diabe", "r10lunge", "r10psych",
        "r10arthre", "r10wtresp",
    ]
    wave_cols = []
    for w in HRS_WAVES:
        wave_cols += [
            f"inw{w}", f"r{w}hosp", f"r{w}cesd", f"r{w}dressa",
            f"r{w}batha", f"r{w}eata", f"r{w}beda", f"r{w}toilta",
            f"r{w}walkra", f"r{w}hearte", f"r{w}stroke", f"r{w}cancre",
        ]
    h = pd.read_stata(
        HRS, columns=list(dict.fromkeys(baseline + wave_cols)),
        convert_categoricals=False, preserve_dtypes=False,
    )
    h["ID"] = h["hhidpn"].round().astype("Int64").astype(str)
    h = h[
        h["inw10"].eq(1) & h["r10agey_b"].ge(50) & h["r10hosp"].eq(0)
    ].copy()

    h["female"] = h["ragender"].eq(2).astype(float)
    h["hispanic"] = h["rahispan"].eq(1).astype(float)
    h["black_nonhispanic"] = (
        h["raracem"].eq(2) & ~h["rahispan"].eq(1)
    ).astype(float)
    h["high_school_plus"] = h["raeduc"].ge(3).astype(float)
    h["married_partnered"] = h["r10mstat"].isin([1, 2, 3]).astype(float)
    h.loc[~h["r10bmi"].between(10, 60), "r10bmi"] = np.nan
    h["bmi"] = h["r10bmi"]
    h["ever_smoke"] = h["r10smokev"].where(h["r10smokev"].isin([0, 1]))
    h["current_smoke"] = h["r10smoken"].where(h["r10smoken"].isin([0, 1]))
    h["current_drink"] = h["r10drink"].where(h["r10drink"].isin([0, 1]))
    base_diseases = [
        "r10hibpe", "r10diabe", "r10lunge", "r10hearte", "r10stroke",
        "r10cancre", "r10psych", "r10arthre",
    ]
    for col in base_diseases:
        h[col] = h[col].where(h[col].isin([0, 1]))
    h["comorbidity"] = h[base_diseases].sum(axis=1, min_count=5)
    h["survey_weight"] = h["r10wtresp"]
    positive = h.loc[h["survey_weight"].gt(0), "survey_weight"]
    if len(positive):
        lo, hi = positive.quantile([0.01, 0.99])
        h["survey_weight"] = h["survey_weight"].clip(lo, hi)

    keep_base = [
        "ID", "r10agey_b", "female", "hispanic", "black_nonhispanic",
        "high_school_plus", "married_partnered", "bmi", "ever_smoke",
        "current_smoke", "current_drink", "comorbidity", "survey_weight",
        "radyear",
    ]
    frames = []
    for w in HRS_WAVES:
        x = h[[
            "ID", f"inw{w}", f"r{w}hosp", f"r{w}cesd", f"r{w}dressa",
            f"r{w}batha", f"r{w}eata", f"r{w}beda", f"r{w}toilta",
            f"r{w}walkra", f"r{w}hearte", f"r{w}stroke", f"r{w}cancre",
        ]].copy()
        x = x.rename(columns={
            f"inw{w}": "inw", f"r{w}hosp": "hospital",
            f"r{w}cesd": "cesd", f"r{w}dressa": "dress",
            f"r{w}batha": "bathe", f"r{w}eata": "eat",
            f"r{w}beda": "bed", f"r{w}toilta": "toilet",
            f"r{w}walkra": "walk_room", f"r{w}hearte": "heart",
            f"r{w}stroke": "stroke", f"r{w}cancre": "cancer",
        })
        for col in ["hospital", *COMMON_ITEMS, "walk_room", "heart", "stroke", "cancer"]:
            x[col] = x[col].where(x[col].isin([0, 1]))
        x["adl5"] = x[COMMON_ITEMS].sum(axis=1, min_count=5)
        x["adl6_native"] = x[[*COMMON_ITEMS, "walk_room"]].sum(axis=1, min_count=6)
        x["adl5_any"] = np.where(x["adl5"].notna(), x["adl5"].gt(0).astype(float), np.nan)
        x["wave"] = w
        x["year"] = HRS_YEARS[w]
        x = x.merge(h[keep_base], on="ID", how="left", validate="many_to_one")
        frames.append(x)
    a = pd.concat(frames, ignore_index=True).sort_values(["ID", "wave"]).reset_index(drop=True)

    for disease in ["heart", "stroke", "cancer"]:
        prev = a.groupby("ID")[disease].shift(1)
        a[f"new_{disease}"] = a[disease].eq(1) & prev.eq(0)
    a["new_major"] = a[["new_heart", "new_stroke", "new_cancer"]].any(axis=1)

    incident = (
        a[a["wave"].gt(10) & a["hospital"].eq(1)]
        .sort_values(["ID", "wave"]).drop_duplicates("ID")[["ID", "wave"]]
        .rename(columns={"wave": "incident_wave"})
    )
    a = a.merge(incident, on="ID", how="left")
    major_ids = set(a.loc[
        a["incident_wave"].notna() & a["wave"].eq(a["incident_wave"]) & a["new_major"], "ID"
    ])
    a["major_at_hospitalization"] = a["ID"].isin(major_ids)
    a["died_by_wave"] = a["radyear"].notna() & a["radyear"].le(a["year"])
    a["death_or_adl"] = np.where(
        a["died_by_wave"], 1.0,
        np.where(a["inw"].eq(1) & a["adl5_any"].notna(), a["adl5_any"], np.nan),
    )
    base = a[a["wave"].eq(10)][["ID", "adl5", "cesd"]].rename(
        columns={"adl5": "base_adl5", "cesd": "base_cesd"}
    )
    a = a.merge(base, on="ID", how="left", validate="many_to_one")
    return h, a


def load_charls_common5():
    roots = {"dress": "dressa", "bathe": "batha", "eat": "eata", "bed": "beda", "toilet": "toilta"}
    cols = ["ID", "r1agey", "inw1", "inw2", "inw3", "inw4"]
    for w in CHARLS_WAVES:
        cols += [f"r{w}hosp1y", *[f"r{w}{root}" for root in roots.values()]]
    h = pd.read_stata(
        CHARLS, columns=list(dict.fromkeys(cols)), convert_categoricals=False,
        preserve_dtypes=False,
    )
    h["ID"] = h["ID"].astype(str)
    h = h[h["inw1"].eq(1) & h["r1agey"].ge(50) & h["r1hosp1y"].eq(0)].copy()
    frames = []
    for w in CHARLS_WAVES:
        x = h[["ID", f"inw{w}", f"r{w}hosp1y", *[f"r{w}{r}" for r in roots.values()]]].copy()
        rename = {f"inw{w}": "inw", f"r{w}hosp1y": "hospital"}
        rename.update({f"r{w}{root}": item for item, root in roots.items()})
        x = x.rename(columns=rename)
        for col in ["hospital", *COMMON_ITEMS]:
            x[col] = x[col].where(x[col].isin([0, 1]))
        x["adl5"] = x[COMMON_ITEMS].sum(axis=1, min_count=5)
        x["wave"] = w
        x["year"] = CHARLS_YEARS[w]
        frames.append(x)
    a = pd.concat(frames, ignore_index=True).sort_values(["ID", "wave"]).reset_index(drop=True)
    incident = (
        a[a["wave"].gt(1) & a["hospital"].eq(1)]
        .sort_values(["ID", "wave"]).drop_duplicates("ID")[["ID", "wave"]]
        .rename(columns={"wave": "incident_wave"})
    )
    a = a.merge(incident, on="ID", how="left")
    return a


def add_ipcw(a, outcome):
    result = pd.Series(np.nan, index=a.index, dtype=float)
    baseline_cols = [
        "r10agey_b", "female", "hispanic", "black_nonhispanic",
        "high_school_plus", "married_partnered", "bmi", "ever_smoke",
        "current_smoke", "current_drink", "comorbidity", f"base_{outcome}",
    ]
    for wave in HRS_WAVES:
        idx = a["wave"].eq(wave)
        z = a.loc[idx, baseline_cols].copy()
        observed = (a.loc[idx, "inw"].eq(1) & a.loc[idx, outcome].notna()).astype(float)
        if wave == 10:
            result.loc[idx & a[outcome].notna()] = 1.0
            continue
        for col in z.columns:
            z[col] = z[col].fillna(z[col].median())
        X = sm.add_constant(z.astype(float), has_constant="add")
        fit = sm.GLM(observed.to_numpy(), X.to_numpy(), family=sm.families.Binomial()).fit()
        probability = np.clip(fit.predict(X.to_numpy()), 0.03, 0.995)
        stabilized = observed.mean() / probability
        obs_weights = stabilized[observed.to_numpy().astype(bool)]
        if len(obs_weights):
            lo, hi = np.quantile(obs_weights, [0.01, 0.99])
            stabilized = np.clip(stabilized, lo, hi)
        result.loc[idx] = np.where(observed.to_numpy().astype(bool), stabilized, np.nan)
    return result


def make_stack(a, cohorts, mode="not_yet"):
    frames, counts = [], []
    never_ids = set(a.loc[a["incident_wave"].isna(), "ID"])
    for cohort in cohorts:
        cases = set(a.loc[a["incident_wave"].eq(cohort), "ID"])
        controls = never_ids if mode == "never" else set(
            a.loc[a["incident_wave"].isna() | a["incident_wave"].gt(cohort), "ID"]
        )
        z = a[a["ID"].isin(cases | controls)].copy()
        z["cohort"] = cohort
        z["case"] = z["ID"].isin(cases).astype(float)
        if mode == "not_yet":
            z = z[~(
                z["case"].eq(0) & z["incident_wave"].notna() & z["wave"].ge(z["incident_wave"])
            )].copy()
        z["event"] = z["wave"] - cohort
        z["pre2"] = (z["case"].eq(1) & z["event"].le(-2)).astype(float)
        z["index"] = (z["case"].eq(1) & z["event"].eq(0)).astype(float)
        z["post1"] = (z["case"].eq(1) & z["event"].eq(1)).astype(float)
        z["post2"] = (z["case"].eq(1) & z["event"].ge(2)).astype(float)
        z["person_fe"] = z["ID"] + "_c" + str(cohort)
        z["wave_fe"] = "c" + str(cohort) + "_w" + z["wave"].astype(str)
        frames.append(z)
        counts.append({
            "cohort_wave": cohort, "cases": len(cases), "controls_at_risk": len(controls),
            "control_mode": mode,
        })
    return pd.concat(frames, ignore_index=True), pd.DataFrame(counts)


def fit_fe(stack, outcome, sensitivity="main", weight_col=None, require_waves=None, include_deaths=False):
    if include_deaths:
        z = stack[stack[outcome].notna()].copy()
    else:
        z = stack[stack["inw"].eq(1) & stack[outcome].notna()].copy()
    if weight_col:
        z = z[z[weight_col].notna() & z[weight_col].gt(0)].copy()
    panel_n = z.groupby("person_fe").size()
    threshold = require_waves if require_waves else 2
    z = z[z["person_fe"].isin(panel_n[panel_n.ge(threshold)].index)].copy()
    ids = np.column_stack([pd.factorize(z["person_fe"])[0], pd.factorize(z["wave_fe"])[0]])
    algorithm = pyhdfe.create(ids, drop_singletons=False, residualize_method="map")
    y0 = z[[outcome]].to_numpy(float)
    X0 = z[EVENT_TERMS].to_numpy(float)
    weights = None if weight_col is None else z[[weight_col]].to_numpy(float)
    y = algorithm.residualize(y0, weights=weights).ravel()
    X = algorithm.residualize(X0, weights=weights)
    keep = np.nanstd(X, axis=0) > 1e-12
    if not keep.any():
        return []
    if weights is None:
        fit = sm.OLS(y, X[:, keep]).fit(cov_type="cluster", cov_kwds={"groups": z["ID"]})
    else:
        fit = sm.WLS(y, X[:, keep], weights=weights.ravel()).fit(
            cov_type="cluster", cov_kwds={"groups": z["ID"]}
        )
    rows, j = [], 0
    for i, term in enumerate(EVENT_TERMS):
        if not keep[i]:
            continue
        beta, se, p = fit.params[j], fit.bse[j], fit.pvalues[j]
        j += 1
        rows.append({
            "sensitivity": sensitivity, "outcome": outcome, "term": term,
            "n_rows": len(z), "n_unique_people": z["ID"].nunique(),
            "estimate": beta, "std_error": se, "ci_low": beta - 1.96 * se,
            "ci_high": beta + 1.96 * se, "p_value": p,
        })
    return rows


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


def make_table1(a):
    b = a[a["wave"].eq(10)].copy()
    b["group"] = np.where(b["incident_wave"].isin(HRS_EVENT_COHORTS), "Incident hospitalization", "At-risk comparison")
    variables = [
        ("Age, years", "r10agey_b", False), ("Female", "female", True),
        ("Hispanic", "hispanic", True), ("Black, non-Hispanic", "black_nonhispanic", True),
        ("High school graduate or above", "high_school_plus", True),
        ("Married/partnered", "married_partnered", True), ("BMI, kg/m2", "bmi", False),
        ("Ever smoked", "ever_smoke", True), ("Current smoker", "current_smoke", True),
        ("Current drinking", "current_drink", True), ("Chronic disease count", "comorbidity", False),
        ("Common five-item ADL difficulty count", "adl5", False), ("CESD-8 score", "cesd", False),
    ]
    case, control = b[b["group"].eq("Incident hospitalization")], b[b["group"].eq("At-risk comparison")]
    rows = []
    for label, col, binary in variables:
        if binary:
            ctext = f"{int(case[col].eq(1).sum())} ({100 * case[col].mean():.1f}%)"
            ntext = f"{int(control[col].eq(1).sum())} ({100 * control[col].mean():.1f}%)"
        else:
            ctext = f"{case[col].mean():.2f} ({case[col].std():.2f})"
            ntext = f"{control[col].mean():.2f} ({control[col].std():.2f})"
        rows.append({
            "Characteristic": label, "Incident hospitalization": ctext,
            "At-risk comparison": ntext,
            "Absolute standardized difference": abs(standardized_difference(case[col], control[col], binary)),
        })
    return pd.DataFrame(rows)


def random_effects_meta(df):
    rows = []
    for term, z in df.groupby("term"):
        z = z.dropna(subset=["estimate", "std_error"])
        if len(z) < 2:
            continue
        yi, vi = z["estimate"].to_numpy(float), z["std_error"].to_numpy(float) ** 2
        wi = 1 / vi
        fixed = np.sum(wi * yi) / np.sum(wi)
        q = np.sum(wi * (yi - fixed) ** 2)
        df_q = len(yi) - 1
        c = np.sum(wi) - np.sum(wi ** 2) / np.sum(wi)
        tau2 = max(0.0, (q - df_q) / c) if c > 0 else 0.0
        wr = 1 / (vi + tau2)
        pooled = np.sum(wr * yi) / np.sum(wr)
        se = np.sqrt(1 / np.sum(wr))
        i2 = max(0.0, 100 * (q - df_q) / q) if q > 0 else 0.0
        rows.append({
            "term": term, "k": len(yi), "random_effect_estimate": pooled,
            "std_error": se, "ci_low": pooled - 1.96 * se, "ci_high": pooled + 1.96 * se,
            "p_value": 2 * norm.sf(abs(pooled / se)), "tau_squared": tau2,
            "Q": q, "Q_p_value": chi2.sf(q, df_q), "I_squared_pct": i2,
        })
    return pd.DataFrame(rows)


def run():
    h, a = load_hrs()
    a["ipcw_adl5"] = add_ipcw(a, "adl5")
    main_stack, counts = make_stack(a, HRS_EVENT_COHORTS, "not_yet")
    never_stack, never_counts = make_stack(a, HRS_EVENT_COHORTS, "never")

    results = []
    for outcome in ["adl5", "adl6_native", "cesd"]:
        results += fit_fe(main_stack, outcome, "main")
        results += fit_fe(never_stack, outcome, "never_controls")
        results += fit_fe(main_stack, outcome, "complete_five_waves", require_waves=5)
        results += fit_fe(main_stack[main_stack["r10agey_b"].ge(60)], outcome, "age_60_plus")
        results += fit_fe(
            main_stack[~(main_stack["case"].eq(1) & main_stack["major_at_hospitalization"])],
            outcome, "exclude_major_disease_onset",
        )
        results += fit_fe(main_stack, outcome, "survey_weighted", "survey_weight")
        if outcome == "adl5":
            results += fit_fe(main_stack, outcome, "ipcw", "ipcw_adl5")
            results += fit_fe(main_stack[main_stack["cohort"].eq(12)], outcome, "middle_2014_cohort")
    results += fit_fe(main_stack, "adl5_any", "binary_any_adl")
    results += fit_fe(main_stack, "death_or_adl", "death_or_any_adl", include_deaths=True)
    result_df = pd.DataFrame(results)

    item_rows = []
    for item in COMMON_ITEMS:
        item_rows += fit_fe(main_stack, item, "adl_item")
    item_df = pd.DataFrame(item_rows)
    item_df["q_value_bh"] = np.nan
    tested = item_df["term"].isin(["post1", "post2"])
    item_df.loc[tested, "q_value_bh"] = bh_adjust(item_df.loc[tested, "p_value"])

    cohort_rows = []
    for wave in HRS_EVENT_COHORTS:
        z = main_stack[main_stack["cohort"].eq(wave)]
        cohort_rows += fit_fe(z, "adl5", f"{HRS_YEARS[wave]} incident cohort")
        cohort_rows += fit_fe(z, "cesd", f"{HRS_YEARS[wave]} incident cohort")
    cohort_df = pd.DataFrame(cohort_rows)

    subgroup_rows = []
    for label, z in [
        ("Age 50-59", main_stack[main_stack["r10agey_b"].lt(60)]),
        ("Age 60+", main_stack[main_stack["r10agey_b"].ge(60)]),
        ("Men", main_stack[main_stack["female"].eq(0)]),
        ("Women", main_stack[main_stack["female"].eq(1)]),
    ]:
        subgroup_rows += fit_fe(z, "adl5", label)
    subgroup_df = pd.DataFrame(subgroup_rows)

    charls = load_charls_common5()
    charls_stack, charls_counts = make_stack(charls, CHARLS_EVENT_COHORTS, "not_yet")
    charls_results = pd.DataFrame(fit_fe(charls_stack, "adl5", "CHARLS common-five age50+"))
    hrs_main = result_df[(result_df["sensitivity"].eq("main")) & result_df["outcome"].eq("adl5")].copy()
    hrs_main["study"] = "HRS"
    charls_results["study"] = "CHARLS"
    cross = pd.concat([charls_results, hrs_main], ignore_index=True)
    baseline_sd = {
        "HRS": a.loc[a["wave"].eq(10), "adl5"].std(),
        "CHARLS": charls.loc[charls["wave"].eq(1), "adl5"].std(),
    }
    cross["baseline_sd"] = cross["study"].map(baseline_sd)
    for col in ["estimate", "std_error", "ci_low", "ci_high"]:
        cross[f"standardized_{col}"] = cross[col] / cross["baseline_sd"]
    meta = random_effects_meta(cross[["study", "term", "estimate", "std_error"]])

    comparison_rows = []
    for term in EVENT_TERMS:
        hc = cross[(cross["study"].eq("HRS")) & cross["term"].eq(term)]
        cc = cross[(cross["study"].eq("CHARLS")) & cross["term"].eq(term)]
        if hc.empty or cc.empty:
            continue
        hr, cr = hc.iloc[0], cc.iloc[0]
        difference = hr["estimate"] - cr["estimate"]
        se = np.sqrt(hr["std_error"] ** 2 + cr["std_error"] ** 2)
        comparison_rows.append({
            "term": term, "HRS_minus_CHARLS": difference, "std_error": se,
            "ci_low": difference - 1.96 * se, "ci_high": difference + 1.96 * se,
            "p_difference": 2 * norm.sf(abs(difference / se)),
        })
    comparison = pd.DataFrame(comparison_rows)

    baseline = a[a["wave"].eq(10)]
    flow = pd.DataFrame([
        {"stage": "HRS 2010 respondents aged 50+ with no hospitalization since prior interview", "n": baseline["ID"].nunique()},
        *[
            {"stage": f"First-observed hospitalization in {HRS_YEARS[w]}", "n": int(baseline["incident_wave"].eq(w).sum())}
            for w in [11, 12, 13, 14]
        ],
        {"stage": "No hospitalization observed through 2018", "n": int(baseline["incident_wave"].isna().sum())},
    ])
    attrition = []
    for cohort in HRS_EVENT_COHORTS:
        ids = set(baseline.loc[baseline["incident_wave"].eq(cohort), "ID"])
        for offset, wave in enumerate(range(cohort, min(14, cohort + 2) + 1)):
            z = a[a["ID"].isin(ids) & a["wave"].eq(wave)]
            attrition.append({
                "cohort_year": HRS_YEARS[cohort], "relative_wave": offset,
                "survey_year": HRS_YEARS[wave], "cohort_n": len(ids),
                "adl5_observed_n": int(z["adl5"].notna().sum()),
                "cesd_observed_n": int(z["cesd"].notna().sum()),
                "died_by_wave_n": int(z["died_by_wave"].sum()),
            })
    attrition_df = pd.DataFrame(attrition)

    table1 = make_table1(a)
    summary = pd.DataFrame([{
        "eligible_HRS_2010_n": baseline["ID"].nunique(),
        "incident_2012_n": int(baseline["incident_wave"].eq(11).sum()),
        "incident_2014_n": int(baseline["incident_wave"].eq(12).sum()),
        "incident_2016_n": int(baseline["incident_wave"].eq(13).sum()),
        "incident_2018_n": int(baseline["incident_wave"].eq(14).sum()),
        "baseline_HRS_adl5_mean": baseline["adl5"].mean(),
        "baseline_HRS_adl5_sd": baseline_sd["HRS"],
        "baseline_CHARLS_adl5_sd": baseline_sd["CHARLS"],
    }])

    result_df.to_csv(OUT / "Table_2_HRS_event_study_results.csv", index=False, encoding="utf-8-sig")
    table1.to_csv(OUT / "Table_1_HRS_baseline_characteristics.csv", index=False, encoding="utf-8-sig")
    item_df.to_csv(OUT / "Table_3_HRS_ADL_item_results.csv", index=False, encoding="utf-8-sig")
    subgroup_df.to_csv(OUT / "Table_4_HRS_subgroup_results.csv", index=False, encoding="utf-8-sig")
    cohort_df.to_csv(OUT / "Table_5_HRS_cohort_specific_results.csv", index=False, encoding="utf-8-sig")
    cross.to_csv(OUT / "Table_6_CHARLS_HRS_harmonized_results.csv", index=False, encoding="utf-8-sig")
    meta.to_csv(OUT / "Table_7_two_cohort_meta_analysis.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "Table_8_cross_cohort_differences.csv", index=False, encoding="utf-8-sig")
    pd.concat([counts, never_counts], ignore_index=True).to_csv(OUT / "cohort_counts.csv", index=False, encoding="utf-8-sig")
    charls_counts.to_csv(OUT / "CHARLS_common5_cohort_counts.csv", index=False, encoding="utf-8-sig")
    flow.to_csv(OUT / "HRS_cohort_flow.csv", index=False, encoding="utf-8-sig")
    attrition_df.to_csv(OUT / "HRS_attrition_by_event_cohort.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "study_summary.csv", index=False, encoding="utf-8-sig")
    cross.to_csv(OUT / "Source_Data_Figure_1.csv", index=False, encoding="utf-8-sig")
    result_df[
        result_df["outcome"].eq("adl5") & result_df["sensitivity"].isin([
            "main", "never_controls", "complete_five_waves", "age_60_plus",
            "exclude_major_disease_onset", "survey_weighted", "ipcw", "middle_2014_cohort",
        ])
    ].to_csv(OUT / "Source_Data_Figure_2.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print("\nHRS main common-five ADL results")
    print(hrs_main.to_string(index=False))
    print("\nHRS sensitivity post results")
    print(result_df[
        result_df["outcome"].eq("adl5") & result_df["term"].isin(["pre2", "post1", "post2"])
    ].to_string(index=False))
    print("\nHarmonized CHARLS-HRS results")
    print(cross.to_string(index=False))


if __name__ == "__main__":
    run()
