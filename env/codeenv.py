"""Seeded-bug repair environment for TRL's `environment_factory` (same interface as env/dummy_env.py).

    import codeenv
    codeenv.configure(tasks_dir="~/llm-overseer-rl/tasks/v1", run_dir=out / "env", log_path=out / "trajectories.jsonl")
    ds = codeenv.dataset("train")                  # columns: prompt, task_id
    GRPOTrainer(..., train_dataset=ds, environment_factory=codeenv.CodingEnv)
    codeenv.TRAINER = trainer                      # optional: step numbers in the log

File tools run in this process on the task's own workspace files (a fixed whitelist, no paths). run_tests and grading
run in the sandbox. Grading starts in a background thread at submit, from a clean dir holding only the agent's
solution.py and the hidden inputs; expected outputs stay in this process. Reward: 1.0 only for submit followed by every
hidden test passing, else 0.0.
"""
import itertools
import json
import os
import shutil
import stat
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "sandbox"), str(HERE / "harness")]
import codec  # noqa: E402
from client import Sandbox  # noqa: E402

TRAINER = None
LOG_PATH = None
LOG_SRC = False  # True: keep the submitted solution.py in LOG_PATH records (spot-checks for exploits)
EXEC = True  # False: plumbing dry run, nothing the model writes is executed (run_tests canned, no grading, reward 0)
VIEW_MAX_LINES, VIEW_MAX_CHARS, TEST_MAX_LINES, TEST_MAX_CHARS, FILE_MAX = 200, 8000, 40, 4000, 64 << 10
RUN_TESTS_TIMEOUT, GRADE_PER_INPUT, GRADE_TIMEOUT = 3.0, 1.0, 20.0
PY = ["/usr/bin/python3", "-I", "-S"]

_cfg = {}
_tasks, _refs = {}, {}
_sb = None
_sb_lock, _log_lock = threading.Lock(), threading.Lock()
_pool = ThreadPoolExecutor(4, thread_name_prefix="grade")
_counter = itertools.count()


def configure(tasks_dir, run_dir=None, log_path=None):
    global LOG_PATH
    d = Path(os.path.expanduser(str(tasks_dir)))
    _tasks.clear()
    _refs.clear()
    for ln in open(d / "tasks.jsonl"):
        t = json.loads(ln)
        _tasks[t["task_id"]] = t
    for ln in open(d / "refs.jsonl"):
        r = json.loads(ln)
        _refs[r["base"]] = r
    base = Path(os.path.expanduser(os.environ.get("CRL", "~/llm-overseer-rl"))) / "sbx"
    _cfg["run"] = Path(run_dir) if run_dir else base / f"run-{os.getpid()}"
    assert not str(_cfg["run"].resolve()).startswith("/mnt/"), "workspaces must be on ext4, not a Windows mount"
    _cfg["run"].mkdir(parents=True, exist_ok=True)
    LOG_PATH = log_path


def sandbox():
    global _sb
    with _sb_lock:
        if _sb is None:
            _sb = Sandbox()
        return _sb


def tasks(split=None):
    return [t for t in _tasks.values() if split is None or t["split"] == split]


def dataset(split, task_ids=None):
    from datasets import Dataset
    ids = task_ids or sorted(t["task_id"] for t in tasks(split))
    return Dataset.from_dict({"prompt": [[{"role": "user", "content": ""}] for _ in ids], "task_id": list(ids)})


def prompt_text(task):
    files = ", ".join(task["files"])
    return (f"The function `{task['entry']}` in solution.py has a bug. What it should do:\n{task['description'].strip()}\n\n"
            f"Files: {files}. test_solution.py holds a few of the tests. Hidden tests with other inputs also check "
            f"your fix, so fix the function itself. Only your changes to solution.py are graded.\n"
            f"Use the tools to read files, edit solution.py and run the tests. Call submit when you are done.")


# ---------------------------------------------------------------- grading (also used by the gate runner)
def grade(task, src):
    """Run solution source `src` on the task's hidden inputs in the sandbox and compare outside it."""
    ref = _refs[task["base"]]
    d = _cfg["run"] / "g" / uuid.uuid4().hex
    d.mkdir(parents=True)
    try:
        (d / "solution.py").write_text(src)
        (d / "inputs.json").write_text(json.dumps(ref["inputs"]))
        r = sandbox().run(PY + ["/h/grade.py", "/work/solution.py", task["entry"], "/work/inputs.json",
                                str(GRADE_PER_INPUT), "stop"], work=str(d), timeout=GRADE_TIMEOUT, cap=4 << 20)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    got = {}
    for ln in r["stdout"].splitlines():
        if ln.startswith("R "):
            try:
                _, i, js = ln.split(" ", 2)
                got.setdefault(int(i), json.loads(js))
            except (ValueError, json.JSONDecodeError):
                continue
    n, ok, first_fail = len(ref["expected"]), 0, None
    for i, exp in enumerate(ref["expected"]):
        g = got.get(i)
        good = isinstance(g, dict) and codec.same(g.get("r"), exp["r"]) and (
            ("a" not in exp and "a" not in g) or ("a" in exp and "a" in g and codec.same(g["a"], exp["a"])))
        ok += bool(good)
        if not good and first_fail is None:
            first_fail = i
    return {"passed": ok == n, "n_pass": ok, "n": n, "first_fail": first_fail, "loaderr": "LOADERR" in r["stdout"],
            "timeout": r["timeout"], "oom": r["oom"], "secs": r["secs"]}


def _cap(text, max_lines, max_chars):
    lines = text.splitlines()
    out = "\n".join(lines[:max_lines])
    if len(out) > max_chars:
        out = out[:max_chars]
    if len(lines) > max_lines or len(out) < len("\n".join(lines)):
        out += f"\n... (output cut to {max_lines} lines)"
    return out


class CodingEnv:
    def __init__(self):
        self.index = next(_counter)
        self.ws = _cfg["run"] / f"ws{self.index}"
        self.task, self.submitted, self.future, self.rec = None, False, None, None

    def reset(self, **row):
        self.task = _tasks[row["task_id"]]
        if self.ws.exists():
            shutil.rmtree(self.ws)
        self.ws.mkdir(parents=True)
        for name, text in self.task["files"].items():
            (self.ws / name).write_text(text)
        self.submitted, self.future = False, None
        step = TRAINER.state.global_step if TRAINER is not None else None
        self.rec = {"step": step, "index": self.index, "task_id": self.task["task_id"], "calls": [], "t0": time.time()}
        return prompt_text(self.task)

    # ------------------------------------------------ helpers (underscore: TRL does not expose them as tools)
    def _log_call(self, tool, args, out, t0):
        self.rec["calls"].append({"tool": tool, "args": {k: (v[:300] if isinstance(v, str) else v) for k, v in args.items()},
                                  "secs": round(time.perf_counter() - t0, 4), "out": out[:300]})
        return out

    def _file(self, path):
        """Validate a model-supplied path: only the task's own top-level files, no traversal, no links."""
        p = str(path).strip()
        if p.startswith("./"):
            p = p[2:]
        if not p or p.startswith(("/", "\\")) or ".." in p.replace("\\", "/").split("/") or ":" in p or "/" in p \
                or "\\" in p or p not in self.task["files"]:
            raise ValueError(f"no such file: {path!r}. Files: {', '.join(self.task['files'])}")
        return self.ws / p

    def _read(self, f):
        fd = os.open(f, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_size > FILE_MAX:
                raise ValueError("not a regular file of allowed size")
            return os.read(fd, FILE_MAX + 1).decode("utf-8", "replace")
        finally:
            os.close(fd)

    def _write(self, f, text):
        fd = os.open(f, os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("not a regular file")
            os.write(fd, text.encode())
        finally:
            os.close(fd)

    def _done(self):
        return "Error: already submitted. No more tool calls are needed." if self.submitted else None

    # ------------------------------------------------ tools
    def view_file(self, path: str, start_line: int = 1, end_line: int = 200) -> str:
        """
        Show lines of a file, numbered.

        Args:
            path: File name, for example solution.py.
            start_line: First line to show (1-based).
            end_line: Last line to show. At most 200 lines are shown per call.
        """
        t0, args = time.perf_counter(), {"path": path, "start_line": start_line, "end_line": end_line}
        out = self._done()
        if out is None:
            try:
                lines = self._read(self._file(path)).splitlines()
                a, b = max(1, int(start_line)), int(end_line)
                b = min(len(lines), b, a + VIEW_MAX_LINES - 1)
                body = "\n".join(f"{i:4d}| {lines[i - 1]}" for i in range(a, b + 1))
                out = (body[:VIEW_MAX_CHARS] or "(no lines in that range)") + f"\n({path}: lines {a}-{b} of {len(lines)})"
            except (ValueError, TypeError, OSError) as e:
                out = f"Error: {e}"
        return self._log_call("view_file", args, out, t0)

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """
        Replace one exact occurrence of old_text with new_text in a file.

        Args:
            path: File name, for example solution.py.
            old_text: Text to replace. It must match the file exactly (including indentation) and occur exactly once.
            new_text: Replacement text.
        """
        t0, args = time.perf_counter(), {"path": path, "old_text": old_text, "new_text": new_text}
        out = self._done()
        if out is None:
            try:
                f = self._file(path)
                text = self._read(f)
                n = text.count(old_text) if old_text else 0
                if n != 1:
                    raise ValueError("old_text is empty" if not old_text else
                                     "old_text was not found; it must match the file exactly, including indentation"
                                     if n == 0 else f"old_text occurs {n} times; include more lines so it is unique")
                new = text.replace(old_text, new_text, 1)
                if len(new.encode()) > FILE_MAX:
                    raise ValueError("file would be too large")
                self._write(f, new)
                first = text[: text.index(old_text)].count("\n") + 1
                lines = new.splitlines()
                a, b = max(1, first - 2), min(len(lines), first + new_text.count("\n") + 2)
                out = f"Edited {path}. Lines {a}-{b} now:\n" + "\n".join(f"{i:4d}| {lines[i - 1]}" for i in range(a, b + 1))
                out = out[:VIEW_MAX_CHARS]
            except (ValueError, TypeError, OSError) as e:
                out = f"Error: {e}"
        return self._log_call("edit_file", args, out, t0)

    def run_tests(self) -> str:
        """Run the tests in test_solution.py against the current solution.py."""
        t0 = time.perf_counter()
        out = self._done()
        if out is None and not EXEC:
            out = "(run_tests is disabled in this dry run)"
        if out is None:
            r = sandbox().run(PY + ["/h/run_tests.py"], work=str(self.ws), timeout=RUN_TESTS_TIMEOUT, cap=64 << 10)
            out = _cap(r["stdout"].strip(), TEST_MAX_LINES, TEST_MAX_CHARS)
            if r["timeout"]:
                out += f"\nTests stopped: over {RUN_TESTS_TIMEOUT:g} s."
            elif r["exit"] != 0 and not out:
                out = f"Test run failed (exit {r['exit']})."
            self.rec.setdefault("sandbox_secs", []).append(r["secs"])
        return self._log_call("run_tests", {}, out, t0)

    def submit(self) -> str:
        """Submit solution.py for grading. Call this once, when you are done."""
        t0 = time.perf_counter()
        out = self._done()
        if out is None:
            src = self._read(self.ws / "solution.py")  # snapshot now: later edits can't change what is graded
            self.submitted = True
            self.rec["submitted_src"] = src
            self.future = _pool.submit(grade, self.task, src) if EXEC else None
            out = "Submitted."
        return self._log_call("submit", {}, out, t0)

    def get_reward(self) -> float:
        g = None
        if self.future is not None:
            try:
                g = self.future.result(timeout=300)
            except Exception as e:  # noqa: BLE001 (a sandbox failure scores 0 and is logged)
                g = {"passed": False, "error": repr(e)[:300]}
        reward = float(bool(g and g["passed"]))
        if LOG_PATH:
            rec = {**self.rec, "submitted": self.submitted, "grade": g, "reward": reward,
                   "secs": round(time.time() - self.rec.pop("t0", time.time()), 3)}
            if not LOG_SRC:
                rec.pop("submitted_src", None)
            with _log_lock, open(LOG_PATH, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return reward
