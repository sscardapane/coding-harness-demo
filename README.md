# Tasklet

Tasklet is a deliberately small, deliberately broken JSON-backed task tracker.
It is designed as the target repository for a coding-agent exercise.

## Run It

```bash
python3 -m tasklet.cli --db tasks.json add "Write the notebook"
python3 -m tasklet.cli --db tasks.json list
python3 -m tasklet.cli --db tasks.json complete 1
python3 -m tasklet.cli --db tasks.json list --all
```

## Run Tests

```bash
python3 -m unittest discover -s tests
```

## Exercise Goal

The test suite currently fails. Build an agent that can inspect this repository,
run the tests, edit the code, and make the tests pass.

The intended bug is in the task-completion path.
