"""Run the complete CHARLS-HRS statistical analysis and figure pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from config import OUTPUT_DIR, require_raw_data


SCRIPTS = [
    "primary_analysis.py",
    "multidomain_analysis.py",
    "targeted_extensions.py",
    "plot_main_results.py",
    "plot_extensions.py",
    "validate_results.py",
]


def main() -> None:
    require_raw_data()
    script_dir = Path(__file__).resolve().parent
    for script in SCRIPTS:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(script_dir / script)], check=True)
    print(f"Analysis complete. Results were written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
