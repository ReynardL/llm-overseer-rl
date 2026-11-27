"""Dummy coding environment for plumbing checks. Tools return canned strings; no model-generated code ever runs.

Task choice is seeded by (global_step, group), so a resumed run sees the same task IDs.
- global_step: read from TRAINER, set by the runner right after GRPOTrainer(...) is built.
- rollout index: TRL hands out pooled instances in creation order (grpo_trainer.py L2457-2465, probe at L733/L750),
  so instance k always serves rollout k of the per-process generation batch.
"""
import hashlib
import itertools
import json

TRAINER = None  # set by the runner
NUM_GENERATIONS = 4
LOG_PATH = None  # jsonl of (step, index, task_id), written on every reset
_counter = itertools.count()

CANNED_FILE = "def add(a, b):\n    return a - b  # BUG\n"
CANNED_TESTS = "FAILED test_add: assert add(2, 3) == 5\n1 failed, 0 passed"

# Long-rollout mode: tool outputs of a few hundred tokens, and submit is refused before MIN_CALLS tool calls,
# so trajectories approach the 4K budget. Still canned text: nothing the model writes is executed.
LONG = False
MIN_CALLS = 8


def _long_file(path):
    body = "".join(f'def helper_{i}(x, y):\n    """Combine x and y for stage {i} of the pipeline."""\n'
                   f"    return (x * {i + 2} + y) % {97 + i}\n\n" for i in range(12))
    return f"# {path}\nimport math\n\n{body}def add(a, b):\n    return a - b  # BUG\n"


LONG_TESTS = ("".join(f"tests/test_core.py::test_helper_{i} PASSED\n" for i in range(12))
              + "tests/test_core.py::test_add FAILED\n\n=== FAILURES ===\n___ test_add ___\n\n    def test_add():\n"
              + ">       assert add(2, 3) == 5\nE       assert -1 == 5\nE        +  where -1 = add(2, 3)\n\n"
              + "tests/test_core.py:41: AssertionError\n=== short test summary info ===\n"
              + "FAILED tests/test_core.py::test_add - assert -1 == 5\n=== 1 failed, 12 passed in 0.31s ===")


def task_id(step, group):
    return int(hashlib.sha256(f"phase0:{step}:{group}".encode()).hexdigest()[:8], 16) % 10_000


class DummyCodingEnv:
    def __init__(self):
        self.index = next(_counter)
        self.submitted = False
        self.calls = 0

    def reset(self, **kwargs):
        step = TRAINER.state.global_step
        self.task = task_id(step, self.index // NUM_GENERATIONS)
        self.submitted, self.calls = False, 0
        if LOG_PATH:
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps({"step": step, "index": self.index, "task": self.task}) + "\n")
        if LONG:
            return (f"Task {self.task}: a test fails in this repo (files: core.py, utils.py, pipeline.py, add.py, "
                    "tests/test_core.py). Inspect the files and tests with the tools, fix the bug with edit_file, "
                    "rerun the tests, then submit.")
        return (
            f"Task {self.task}: fix the bug in add.py so the tests pass. Use the tools: view_file, run_tests, "
            "edit_file, then submit."
        )

    def view_file(self, path: str) -> str:
        """
        View a file in the repository.

        Args:
            path: Path of the file to view.
        """
        self.calls += 1
        return f"TOOLOUT_VIEW {path}\n{_long_file(path) if LONG else CANNED_FILE}"

    def edit_file(self, path: str, content: str) -> str:
        """
        Replace the whole content of a file.

        Args:
            path: Path of the file to edit.
            content: New file content.
        """
        self.calls += 1
        return f"TOOLOUT_EDIT ok {path} ({len(content)} chars)"

    def run_tests(self) -> str:
        """Run the visible tests."""
        self.calls += 1
        return f"TOOLOUT_TESTS\n{LONG_TESTS if LONG else CANNED_TESTS}"

    def submit(self) -> str:
        """Submit the current repository state for grading."""
        if LONG and self.calls < MIN_CALLS:
            return (f"TOOLOUT_SUBMIT rejected: only {self.calls} tool calls so far, at least {MIN_CALLS} are required. "
                    "Look at the other files and rerun the tests before submitting.")
        self.submitted = True
        return "TOOLOUT_SUBMIT received"

    def get_reward(self) -> float:
        # Pseudo-random but deterministic, so groups have variance and the loss is nonzero.
        h = int(hashlib.sha256(f"{self.task}:{self.index}:{self.calls}".encode()).hexdigest()[:4], 16)
        return float(self.submitted and h % 2 == 0)
