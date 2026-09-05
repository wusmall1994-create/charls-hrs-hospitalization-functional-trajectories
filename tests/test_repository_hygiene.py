"""Repository checks that do not require access to CHARLS or HRS data."""

from __future__ import annotations

import py_compile
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROHIBITED_SUFFIXES = {
    ".dta", ".sav", ".sas7bdat", ".xpt", ".csv", ".xlsx", ".xls",
    ".doc", ".docx", ".pdf", ".png", ".tif", ".tiff", ".svg", ".zip",
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yml", ".yaml", ".json", ".cff"}


class RepositoryHygieneTests(unittest.TestCase):
    def test_python_files_compile(self) -> None:
        for path in ROOT.rglob("*.py"):
            if ".git" not in path.parts:
                py_compile.compile(str(path), doraise=True)

    def test_no_data_manuscript_or_figure_files(self) -> None:
        offenders = [
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_file() and ".git" not in path.parts
            and "results" not in path.relative_to(ROOT).parts
            and "data" not in path.relative_to(ROOT).parts
            and path.suffix.lower() in PROHIBITED_SUFFIXES
        ]
        self.assertEqual(offenders, [])

    def test_no_local_absolute_paths_or_personal_contacts(self) -> None:
        forbidden_literals = [
            "fj" + "doc",
            "fjdoctor" + "liang@163.com",
            "19959" + "313218",
        ]
        windows_path = re.compile(r"(?i)(?:^|[\"'])\s*[a-z]:[\\/]")
        offenders: list[str] = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8")
            if any(value.lower() in text.lower() for value in forbidden_literals):
                offenders.append(path.relative_to(ROOT).as_posix())
            if windows_path.search(text):
                offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(sorted(set(offenders)), [])


if __name__ == "__main__":
    unittest.main()
