"""CPU stand-in for run/trainer.py, for test_orch.py. Same flags, files and exit codes; no model.

A node counts generations from 0; its checkpoint after generation g is checkpoint-<g * U> (U = updates per generation)
and holds a fake adapter weight w and an optimizer step count (json). Each generation adds learning_rate / 2e-5 to w and
writes 32 rows (rows.jsonl), 32 env records (trajectories.jsonl) and U trainer log lines (steps.jsonl), all a function
of (run, stream name, generation, row), so a redone generation repeats itself. Tasks whose hash is 0 mod 5 always get
reward 0 (zero-signal visits). Val pass rate rises with w: task t passes when its fixed hash u_t < p(w), so evals are
paired like real ones (a better checkpoint passes a superset).
FAKE_SCENARIO (json file) injects events, each fired once per run dir (marker files); keys are node:generation count:
  {"pause_train": ["n0:2"], "pause_eval": ["n0:3"], "fail": ["n1:2"], "ram_kill": ["n0:5"], "fail_after_eval": ["n0:3"], "flat": true}
pause_train touches the pause file at the end of that generation (as the operator would); the trainer then saves and exits 3.
pause_eval touches it halfway through the eval after that generation (partial traj, no done.json, exit 3).
fail exits 1 before training that generation (an OOM, say); ram_kill exits 97 there with watch.sh's host-RAM kill line;
fail_after_eval exits 1 after the eval at that generation wrote done.json. flat: val never moves (the A2 gate refuses).
logps_fallback logs a rollout-logps fallback count of 1 at that generation (the fake trainer itself goes on).
Every launch's argv goes to <out>/fake_launches.jsonl.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
for f in ("--model", "--vllm-util", "--mem-frac", "--env", "--tasks", "--node-config", "--stream", "--out", "--val-tasks",
          "--val-out", "--pause-file", "--init-adapter", "--init-optimizer"):
    ap.add_argument(f)
for f in ("--max-gens", "--save-gens", "--stop-at-gen", "--seed"):
    ap.add_argument(f, type=int, default=0)
for f in ("--unsloth", "--vllm", "--resume", "--eval-only", "--allow-test"):
    ap.add_argument(f, action="store_true")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
node = out.parent.name
with open(out / "fake_launches.jsonl", "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
sc = json.loads(Path(os.environ["FAKE_SCENARIO"]).read_text()) if os.environ.get("FAKE_SCENARIO") else {}
cfg = json.loads(Path(a.node_config).read_text())
stream = json.loads(Path(a.stream).read_text())
pause = Path(a.pause_file)
G = cfg["num_generations"]
P, U = 32 // G, 32 // cfg["rollouts_per_update"]
rows = stream["rows"]
# the real trainer's startup asserts
assert stream["P"] == P and rows and len(rows) % P == 0, (stream["P"], P, len(rows))
assert len(rows) >= a.stop_at_gen * P, f"generations up to --stop-at-gen {a.stop_at_gen} are not all assigned"
assert a.max_gens > 0 and a.save_gens > 0 and 0 <= a.stop_at_gen <= a.max_gens
assert not (a.resume and a.init_adapter), "resume and init-adapter are exclusive"
assert not a.init_optimizer or (a.init_adapter == a.init_optimizer and not cfg["reset_optimizer"] and not a.eval_only)
assert a.eval_only or a.stop_at_gen, "a training launch needs --stop-at-gen"
assert not (a.eval_only and a.resume)


def fire(kind, gen):
    m = out / f".fired_{kind}_{gen}"
    if f"{node}:{gen}" in sc.get(kind, []) and not m.exists():
        m.touch()
        return True
    return False


def ckpts():
    return sorted(int(p.name.split("-")[1]) for p in out.glob("checkpoint-*") if (p / "trainer_state.json").exists())


def save(gen, w, opt):
    d = out / f"checkpoint-{gen * U}"
    d.mkdir(exist_ok=True)
    (d / "adapter.json").write_text(json.dumps({"w": w}))
    (d / "optimizer.json").write_text(json.dumps({"steps": opt}))
    (d / "trainer_state.json").write_text(json.dumps({"global_step": gen * U, "max_steps": a.max_gens * U, "seed": a.seed}))


def load(d):
    return json.loads((Path(d) / "adapter.json").read_text())["w"], json.loads((Path(d) / "optimizer.json").read_text())["steps"]


def summary(gen):  # one line per launch, as the real trainer
    with open(out / "summary.jsonl", "a") as f:
        f.write(json.dumps({"global_step": gen * U, "peak_reserved_GiB": 7.4}) + "\n")


gen, w, opt = 0, 0.0, 0
if a.resume:
    assert ckpts()[-1] % U == 0
    gen = ckpts()[-1] // U
    w, opt = load(out / f"checkpoint-{gen * U}")
elif a.init_adapter and not a.eval_only:
    w, steps = load(a.init_adapter)
    (out / "init_check.json").write_text(json.dumps({"from": a.init_adapter, "w": w, "match": True}))
    if a.init_optimizer:  # the start checkpoint's optimizer state, else a fresh optimizer
        opt = steps
        (out / "init_optimizer_check.json").write_text(json.dumps({"from": a.init_optimizer, "optimizer_steps": opt}))
elif not a.eval_only:
    assert not ckpts(), "fresh start into a dir with checkpoints"

if not a.eval_only:
    lr, idx = cfg["learning_rate"], 0  # idx: the env counter, from 0 in every launch
    for g in range(gen, a.stop_at_gen):
        if fire("fail", g + 1):
            print("FAKE FAIL before generation", g + 1)
            sys.exit(1)
        if fire("ram_kill", g + 1):  # as watch.sh's host-RAM guard: its line in the log, exit 97
            print("WATCHDOG_KILL host free RAM 3.8 GB")
            sys.exit(97)
        rr, tt = [], []
        for j in range(32):
            t = rows[g * P + j // G]
            rng = random.Random(f"{stream['run']}|{stream['name']}|{g}|{j}")
            dead = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) % 5 == 0
            kind = "nosubmit" if dead else rng.choice(["pass", "partial", "wrong", "nosubmit", "loaderr", "timeout"])
            reward = {"pass": 1.0, "partial": 0.25}.get(kind, 0.0)
            grade = None if kind == "nosubmit" else {"passed": kind == "pass", "n_pass": {"pass": 5, "partial": 4}.get(kind, 2), "n": 5,
                                                     "first_fail": None, "loaderr": "SyntaxError" if kind == "loaderr" else None,
                                                     "timeout": kind == "timeout", "oom": False, "secs": 0.1}
            tt.append({"step": g * U, "index": idx, "task_id": t, "calls": [{"name": "view_file"}] * (1 + j % 3), "t0": 0.0,
                       "submitted": kind != "nosubmit", "grade": grade, "passed": float(kind == "pass"), "reward": reward, "secs": 1.0})
            rr.append({"step": g * U, "gen": g, "row": j, "task_id": t, "prompt": 500, "completion": 800 + 10 * (j % 4),
                       "trained_tokens": 0 if j == 5 else 800, "masked_truncated": j == 5, "advantage": 0.0, "reward": reward,
                       "passed": float(kind == "pass"), "env_index": idx})
            idx += 1
        with open(out / "trajectories.jsonl", "a") as f:
            f.write("".join(json.dumps(x) + "\n" for x in tt))
        with open(out / "rows.jsonl", "a") as f:
            f.write("".join(json.dumps(x) + "\n" for x in rr))
        fb = int(fire("logps_fallback", g + 1))  # one batch fell back to trainer-computed old logprobs
        with open(out / "steps.jsonl", "a") as f:
            for u in range(U):  # the first update's line carries the reward stats, as with the real trainer
                rec = {"step": g * U + u + 1, "loss": 0.01 * (u + 1), "grad_norm": 0.1 * (u + 1), "learning_rate": lr,
                       "clip_ratio/region_mean": 0.05 * u, "rollout_logps_as_old_batches": 32 * (g + 1), "rollout_logps_as_old_fallbacks": fb}
                f.write(json.dumps(rec if u else {**rec, "reward": sum(x["reward"] for x in rr) / 32}) + "\n")
        w += lr / 2e-5
        opt += U
        print(f"STEP {(g + 1) * U}", flush=True)
        if fire("pause_train", g + 1):
            pause.touch()
        if pause.exists():
            save(g + 1, w, opt)
            summary(g + 1)
            print("PAUSE seen at step", (g + 1) * U)
            sys.exit(3)
        if (g + 1) % a.save_gens == 0 or g + 1 == a.stop_at_gen:
            save(g + 1, w, opt)
    with open(out / "steps.jsonl", "a") as f:  # HF's end-of-train line: no "loss" key, the same step as the last update
        f.write(json.dumps({"step": a.stop_at_gen * U, "train_runtime": 1.0, "train_loss": 0.0}) + "\n")
    summary(a.stop_at_gen)
    print("SEGMENT_END at step", a.stop_at_gen * U)
    gs = a.stop_at_gen
else:
    gs = int(Path(a.init_adapter).name.split("-")[1]) // U if a.init_adapter else 0
    if a.init_adapter:
        w, _ = load(a.init_adapter)

if not a.val_tasks:
    sys.exit(0)
vo = Path(a.val_out)
assert not (vo / "done.json").exists(), "a finished eval is never overwritten"
if vo.exists():
    n = 1
    while vo.with_name(f"{vo.name}.aborted{n}").exists():
        n += 1
    vo.rename(vo.with_name(f"{vo.name}.aborted{n}"))
vo.mkdir(parents=True)
tasks = [ln.strip() for ln in open(a.val_tasks) if ln.strip()]
p = 0.2 if sc.get("flat") else 0.2 + 0.4 * (1 - 0.97 ** w)
with open(vo / "traj.jsonl", "w") as f:
    for i, t in enumerate(tasks):
        if i == len(tasks) // 2 and fire("pause_eval", gs):
            pause.touch()
        if pause.exists():
            print("EVAL PAUSED")
            sys.exit(3)
        u = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) / 2 ** 32  # per task: a higher w passes a superset (paired evals)
        f.write(json.dumps({"task_id": t, "base": t.rsplit("_", 1)[0], "reward": float(u < p), "end": "stop",
                            "submitted": True, "turns": 3}) + "\n")
(vo / "done.json").write_text(json.dumps({"n": len(tasks), "global_step": gs * U}))
print("VAL", gs, p)
if fire("fail_after_eval", gs):  # a teardown crash after done.json (or the orchestrator killed before it records it)
    print("FAKE FAIL after eval", gs)
    sys.exit(1)
sys.exit(0)
