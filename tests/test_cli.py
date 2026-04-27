from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliTest(unittest.TestCase):
    def run_cli(self, db_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "tasklet.cli", "--db", str(db_path), *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_completed_task_disappears_from_default_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "tasks.json"

            add_result = self.run_cli(db_path, "add", "Fix failing tests")
            self.assertEqual(add_result.returncode, 0, add_result.stderr)

            complete_result = self.run_cli(db_path, "complete", "1")
            self.assertEqual(complete_result.returncode, 0, complete_result.stderr)

            list_result = self.run_cli(db_path, "list")
            self.assertEqual(list_result.returncode, 0, list_result.stderr)
            self.assertIn("No tasks.", list_result.stdout)
            self.assertNotIn("Fix failing tests", list_result.stdout)

    def test_missing_task_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "tasks.json"

            result = self.run_cli(db_path, "complete", "404")

            self.assertEqual(result.returncode, 1)
            self.assertIn("No task with id 404", result.stderr)


if __name__ == "__main__":
    unittest.main()

