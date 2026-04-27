from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .store import TaskStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage a tiny JSON task list.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("tasks.json"),
        help="Path to the JSON task database.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add a new task.")
    add_parser.add_argument("title")

    list_parser = subparsers.add_parser("list", help="List tasks.")
    list_parser.add_argument(
        "--all",
        action="store_true",
        help="Include completed tasks.",
    )

    complete_parser = subparsers.add_parser("complete", help="Mark a task done.")
    complete_parser.add_argument("task_id", type=int)

    delete_parser = subparsers.add_parser("delete", help="Delete a task.")
    delete_parser.add_argument("task_id", type=int)

    return parser


def format_tasks(tasks) -> str:
    if not tasks:
        return "No tasks."

    lines = []
    for task in tasks:
        status = "x" if task.done else " "
        lines.append(f"{task.id}. [{status}] {task.title}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = TaskStore(args.db)

    if args.command == "add":
        task = store.add(args.title)
        print(f"Added task {task.id}: {task.title}")
        return 0

    if args.command == "list":
        print(format_tasks(store.list(include_done=args.all)))
        return 0

    if args.command == "complete":
        try:
            task = store.complete(args.task_id)
        except KeyError:
            print(f"No task with id {args.task_id}", file=sys.stderr)
            return 1
        print(f"Completed task {task.id}: {task.title}")
        return 0

    if args.command == "delete":
        try:
            task = store.delete(args.task_id)
        except KeyError:
            print(f"No task with id {args.task_id}", file=sys.stderr)
            return 1
        print(f"Deleted task {task.id}: {task.title}")
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

