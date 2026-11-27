"""Training reward: a subclass of codeenv.CodingEnv that overrides get_reward only.

Used only by the trainer's rollouts (environment_factory). Eval builds the unmodified codeenv.CodingEnv itself
(eval/evalharness.py run_pool), so eval stays binary. Nothing here patches codeenv.

  full pass                                              -> 1.0
  no submit, no grade, grade error, load error,
  timeout, oom, n missing                                -> 0.0
  otherwise                                              -> CAP * max(0, (n_pass - b) / (n - b)) if n > b else 0.0

b is the unedited buggy file's own n_pass on the hidden tests (null_pass.json, built by build_pools.py), so partial
credit is paid only for hidden tests a submission fixes beyond what the buggy file already passes.

  python3 trainenv.py   self-check (no sandbox)
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "env"))
import codeenv  # noqa: E402

CAP = 0.5  # [decided in shape; the value is a default] the most a submission that does not pass every hidden test can earn
NULL_PASS = {}  # task_id -> the buggy file's n_pass; load() fills it
MISSING = set()  # tasks scored without a table entry (they pay 0 partial credit); the trainer asserts none at startup


def load(path=HERE.parent / "data" / "null_pass.json"):
    NULL_PASS.clear()
    NULL_PASS.update(json.loads(Path(path).read_text()))
    return NULL_PASS


def train_reward(g, b):
    """g: codeenv.grade's dict, a grade error dict, or None (no submit). b: the buggy file's n_pass, or None (not in the table)."""
    if not g:
        return 0.0
    if g.get("passed"):
        return 1.0
    n, k = g.get("n"), g.get("n_pass")
    if g.get("error") or g.get("loaderr") or g.get("timeout") or g.get("oom") or not n or k is None or b is None:
        return 0.0
    return CAP * max(0.0, (k - b) / (n - b)) if n > b else 0.0


class TrainEnv(codeenv.CodingEnv):
    def get_reward(self) -> float:
        g = None
        if self.future is not None:
            try:
                g = self.future.result(timeout=300)
            except Exception as e:  # noqa: BLE001
                g = {"passed": False, "error": repr(e)[:300]}
        tid = self.task["task_id"]
        b = NULL_PASS.get(tid)
        if b is None:
            MISSING.add(tid)
        passed = float(bool(g and g["passed"]))
        reward = train_reward(g, b)
        self.passed, self.reward = passed, reward  # read back by the trainer for rows.jsonl
        if codeenv.LOG_PATH:  # codeenv's record, plus passed (binary) and reward (the training reward)
            rec = {**self.rec, "submitted": self.submitted, "grade": g, "passed": passed, "reward": reward,
                   "secs": round(time.time() - self.rec.pop("t0", time.time()), 3)}
            if not codeenv.LOG_SRC:
                rec.pop("submitted_src", None)
            with codeenv._log_lock, open(codeenv.LOG_PATH, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return reward


if __name__ == "__main__":
    def g(n_pass, n=10, **kw):
        return {"passed": n_pass == n, "n_pass": n_pass, "n": n, "first_fail": None, "loaderr": False, "timeout": False, "oom": 0, **kw}

    assert train_reward(g(10), 4) == 1.0  # full pass
    assert train_reward(g(10), 10) == 1.0 and train_reward(g(10), 99) == 1.0 and train_reward(g(10), None) == 1.0  # even if the table is wrong
    assert train_reward(None, 4) == 0.0  # no submit (or --no-exec: no grade)
    assert train_reward({"passed": False, "error": "TimeoutError()"}, 4) == 0.0  # grade error
    for bad in ({"loaderr": True}, {"timeout": True}, {"oom": 1}, {"n": None}, {"n": 0}, {"n_pass": None}):
        assert train_reward({**g(7), **bad}, 4) == 0.0, bad
    assert train_reward(g(4), 4) == 0.0  # the unchanged buggy file pays 0
    assert train_reward(g(3), 4) == 0.0  # worse than the buggy file
    assert train_reward(g(7), 4) == CAP * 3 / 6 and train_reward(g(9), 4) == CAP * 5 / 6 < CAP
    assert train_reward(g(9), 0) == CAP * 0.9
    assert train_reward(g(7), None) == 0.0  # not in the table: no partial credit
    assert train_reward(g(9, n=10), 10) == 0.0 and train_reward(g(9), 12) == 0.0  # table at or above n: n > b fails
    # the class: get_reward only, with codeenv's log fields plus passed and reward; codeenv.CodingEnv is untouched
    assert [k for k in vars(TrainEnv) if not k.startswith("__")] == ["get_reward"]
    import concurrent.futures
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        codeenv.LOG_PATH = Path(td) / "t.jsonl"
        NULL_PASS.update({"t1": 4})

        def roll(tid, grade):
            e = TrainEnv.__new__(TrainEnv)
            e.task, e.submitted, e.future = {"task_id": tid}, grade is not None, None
            if grade is not None:
                e.future = concurrent.futures.Future()
                e.future.set_result(grade) if isinstance(grade, dict) else e.future.set_exception(grade)
            e.rec = {"step": 0, "index": 0, "task_id": tid, "calls": [], "t0": time.time(), "submitted_src": "x"}
            return e.get_reward()

        assert roll("t1", g(7)) == 0.25 and roll("t1", g(10)) == 1.0 and roll("t1", None) == 0.0 and roll("t1", RuntimeError("x")) == 0.0
        assert not MISSING and roll("t2", g(7)) == 0.0 and MISSING == {"t2"} and roll("t2", g(10)) == 1.0
        recs = [json.loads(ln) for ln in open(codeenv.LOG_PATH)]
        assert [(r["passed"], r["reward"]) for r in recs] == [(0.0, 0.25), (1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 1.0)]
        # t0 is in codeenv's own records too (codeenv spreads self.rec before popping it)
        assert set(recs[0]) == {"step", "index", "task_id", "calls", "t0", "submitted", "grade", "passed", "reward", "secs"}, set(recs[0])
        assert recs[3]["grade"]["error"].startswith("RuntimeError")
        codeenv.LOG_PATH = None
    print("TRAINENV_CHECK ok, CAP", CAP)
