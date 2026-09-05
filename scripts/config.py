"""Portable path configuration for the CHARLS-HRS analysis.

Raw data are never bundled with this repository. Set the three input paths with
environment variables before running the analysis. Generated files are written
to ``results/`` by default and are excluded from version control.
"""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _path_from_environment(name: str, fallback: str) -> Path:
    return Path(os.environ.get(name, fallback)).expanduser().resolve()


CHARLS_DATA = _path_from_environment(
    "CHARLS_DATA", str(REPOSITORY_ROOT / "data" / "H_CHARLS_D_Data.dta")
)
CHARLS_EXIT_DATA = _path_from_environment(
    "CHARLS_EXIT_DATA", str(REPOSITORY_ROOT / "data" / "Exit_Module.dta")
)
HRS_DATA = _path_from_environment(
    "HRS_DATA", str(REPOSITORY_ROOT / "data" / "randhrs1992_2022v1.dta")
)
OUTPUT_DIR = _path_from_environment(
    "CHARLS_HRS_OUTPUT_DIR", str(REPOSITORY_ROOT / "results")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def require_raw_data() -> None:
    """Raise a clear error when a required restricted-use input is absent."""

    missing = [
        ("CHARLS_DATA", CHARLS_DATA),
        ("CHARLS_EXIT_DATA", CHARLS_EXIT_DATA),
        ("HRS_DATA", HRS_DATA),
    ]
    missing = [(name, path) for name, path in missing if not path.is_file()]
    if missing:
        details = "\n".join(f"  {name}: {path}" for name, path in missing)
        raise FileNotFoundError(
            "Required public-use data files were not found. Set these environment "
            f"variables before running the analysis:\n{details}"
        )
