"""Training pools and the null-pass table.

  python3 build_pools.py [--traj ~/llm-overseer-rl/runs/calibration/traj.jsonl] [--pool ~/llm-overseer-rl/tasks/v2]
                         [--null ~/llm-overseer-rl/runs/null_gate/null.jsonl]

--traj is a calibration run of the base model (eval/evalharness.py, 8 samples per train task); k is a task's solved samples.
easy = data/train_pool.txt (k in 1..6), read by path, sha checked; not rewritten.
hard = train-split tasks with all 8 calibration samples and k = 0. Written to hard_pool.txt (sorted ids, one per line).
null_pass.json = {task_id: n_pass of the unedited buggy file on the hidden tests} for the tasks of both pools, from
the null-agent gate run. That file also holds val and test rows: a row is used only if its task id is in a
pool, so no val or test result is read into anything.
Asserts the pools are disjoint from each other, from data/val_tasks.txt and from data/test_tasks.txt (ids only).
Exits 1 if the hard pool has under 100 tasks.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EASY = HERE.parent / "data" / "train_pool.txt"
EASY_SHA = "546e7ea09b396cc718f3f7a7784aebedc3158822ae5519d74c8e31594eba711c"

ap = argparse.ArgumentParser()
ap.add_argument("--traj", default="~/llm-overseer-rl/runs/calibration/traj.jsonl")
ap.add_argument("--pool", default="~/llm-overseer-rl/tasks/v2")
ap.add_argument("--null", default="~/llm-overseer-rl/runs/null_gate/null.jsonl")
a = ap.parse_args()


def ids_of(p):
    return [ln.strip() for ln in open(p) if ln.strip()]


def sha(ids):
    return hashlib.sha256("".join(t + "\n" for t in ids).encode()).hexdigest()


easy = ids_of(EASY)
assert sha(easy) == EASY_SHA, f"data/train_pool.txt sha {sha(easy)}"
split = {}
for ln in open(Path(os.path.expanduser(a.pool)) / "tasks.jsonl"):
    t = json.loads(ln)
    split[t["task_id"]] = (t["split"], t["base"])
ks = defaultdict(list)
for ln in open(os.path.expanduser(a.traj)):
    r = json.loads(ln)
    if split[r["task_id"]][0] == "train":
        ks[r["task_id"]].append((r["sample"], r["reward"]))
for t, v in ks.items():
    assert len({s for s, _ in v}) == len(v), f"duplicate sample in {t}"
hard = sorted(t for t, v in ks.items() if len(v) == 8 and sum(r for _, r in v) == 0)
val, test = set(ids_of(HERE.parent / "data" / "val_tasks.txt")), set(ids_of(HERE.parent / "data" / "test_tasks.txt"))
assert not set(easy) & set(hard), "easy and hard pools overlap"
for name, pool in (("easy", easy), ("hard", hard)):
    assert len(set(pool)) == len(pool), f"duplicate ids in the {name} pool"
    assert not set(pool) & val, f"{name} pool overlaps val"
    assert not set(pool) & test, f"{name} pool overlaps test"
    assert all(split[t][0] == "train" for t in pool), f"{name} pool has a task outside the train split"

both, null = set(easy) | set(hard), {}
for ln in open(os.path.expanduser(a.null)):
    r = json.loads(ln)
    if r["task_id"] not in both:  # val, test and out-of-pool train rows: skipped before any other field is used
        continue
    g = r["grade"]
    assert r["task_id"] not in null and g and isinstance(g["n_pass"], int) and isinstance(g["n"], int) and not g["passed"], r
    assert g["n_pass"] < g["n"], r
    null[r["task_id"]] = g["n_pass"]
missing = sorted(both - set(null))
assert not missing, f"{len(missing)} pool tasks have no null-agent row: {missing[:5]}"

(HERE.parent / "data" / "hard_pool.txt").write_text("".join(t + "\n" for t in hard))
body = json.dumps(dict(sorted(null.items())), separators=(",", ":"))
(HERE.parent / "data" / "null_pass.json").write_text(body)
print(f"EASY {len(easy)} tasks, {len({split[t][1] for t in easy})} functions, sha256 {sha(easy)}")
print(f"HARD {len(hard)} tasks, {len({split[t][1] for t in hard})} functions, sha256 {sha(hard)}")
print(f"NULL_PASS {len(null)} tasks, sha256 {hashlib.sha256(body.encode()).hexdigest()}, "
      f"buggy file passes 0 hidden tests in {sum(v == 0 for v in null.values())}")
print(f"complete train tasks in the calibration run:{sum(len(v) == 8 for v in ks.values())}; val {len(val)}, test {len(test)} ids checked disjoint")
if len(hard) < 100:
    print("HARD POOL UNDER 100 TASKS: stop and report")
    sys.exit(1)
