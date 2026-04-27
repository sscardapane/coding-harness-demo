from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class Task:
    id: int
    title: str
    done: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(id=int(data["id"]), title=str(data["title"]), done=bool(data["done"]))

    def to_dict(self) -> dict:
        return asdict(self)


class TaskStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def list(self, include_done: bool = False) -> list[Task]:
        tasks = self._load()
        if include_done:
            return tasks
        return [task for task in tasks if not task.done]

    def get(self, task_id: int) -> Task:
        for task in self._load():
            if task.id == task_id:
                return task
        raise KeyError(task_id)

    def add(self, title: str) -> Task:
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("task title cannot be empty")

        tasks = self._load()
        task = Task(id=self._next_id(tasks), title=clean_title)
        tasks.append(task)
        self._save(tasks)
        return task

    def complete(self, task_id: int) -> Task:
        tasks = self._load()
        for task in tasks:
            if task.id == task_id:
                task.done = True
                return task
        raise KeyError(task_id)

    def delete(self, task_id: int) -> Task:
        tasks = self._load()
        remaining = []
        deleted = None

        for task in tasks:
            if task.id == task_id:
                deleted = task
            else:
                remaining.append(task)

        if deleted is None:
            raise KeyError(task_id)

        self._save(remaining)
        return deleted

    def _load(self) -> list[Task]:
        if not self.path.exists():
            return []

        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        return [Task.from_dict(item) for item in data]

    def _save(self, tasks: list[Task]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [task.to_dict() for task in tasks]
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")

    def _next_id(self, tasks: list[Task]) -> int:
        if not tasks:
            return 1
        return max(task.id for task in tasks) + 1

