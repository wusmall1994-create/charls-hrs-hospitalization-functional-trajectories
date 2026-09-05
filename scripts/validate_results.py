"""Validate the principal numerical outputs after a complete pipeline run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing expected output files:\n" + "\n".join(missing))


def main() -> None:
    required = [
        OUTPUT_DIR / "Table_2_multidomain_event_study.csv",
        OUTPUT_DIR / "Table_7_cross_cohort_differences.csv",
        OUTPUT_DIR / "Table_7_IADL_cross_cohort_differences.csv",
        OUTPUT_DIR / "Table_S10_ADL_age_linear_interactions.csv",
        OUTPUT_DIR / "Figure_1_CHARLS_HRS_multidomain_replication.png",
        OUTPUT_DIR / "Figure_2_CHARLS_HRS_age_dose_response.png",
    ]
    require_files(required)

    multidomain = pd.read_csv(required[0]).set_index(["domain", "cohort", "term"])
    expected = {
        ("Basic ADL (5 items)", "CHARLS", "post1"): 0.279,
        ("Basic ADL (5 items)", "HRS", "post1"): 0.280,
        ("Instrumental ADL (4 items)", "CHARLS", "post1"): 0.202,
        ("Instrumental ADL (4 items)", "HRS", "post1"): 0.287,
    }
    for key, target in expected.items():
        observed = float(multidomain.loc[key, "standardized_estimate"])
        if abs(observed - target) > 0.002:
            raise AssertionError(f"Unexpected estimate for {key}: {observed:.6f}")

    adl_difference = pd.read_csv(required[1]).set_index("term").loc["post1"]
    if float(adl_difference["p_difference"]) <= 0.95:
        raise AssertionError("The cross-cohort ADL replication check failed")

    iadl_difference = pd.read_csv(required[2]).set_index("term").loc["post1"]
    if float(iadl_difference["p_difference"]) >= 0.05:
        raise AssertionError("The cross-cohort IADL difference check failed")

    print("All principal result checks passed")


if __name__ == "__main__":
    main()
