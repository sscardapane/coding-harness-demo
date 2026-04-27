from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tasklet.store import TaskStore


class TaskStoreTest(unittest.TestCase):
    def test_add_persists_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "tasks.json"
            store = TaskStore(db_path)

            task = store.add("Draft exercise")

            reloaded = TaskStore(db_path)
            self.assertEqual(task.id, 1)
            self.assertEqual(reloaded.get(1).title, "Draft exercise")

    def test_complete_persists_done_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "tasks.json"
            store = TaskStore(db_path)
            store.add("Teach the agent to edit files")

            completed = store.complete(1)

            reloaded = TaskStore(db_path)
            self.assertTrue(completed.done)
            self.assertTrue(reloaded.get(1).done)
            self.assertEqual(reloaded.list(), [])

    def test_delete_persists_removed_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "tasks.json"
            store = TaskStore(db_path)
            store.add("Temporary task")

            deleted = store.delete(1)

            reloaded = TaskStore(db_path)
            self.assertEqual(deleted.title, "Temporary task")
            with self.assertRaises(KeyError):
                reloaded.get(1)


if __name__ == "__main__":
    unittest.main()

