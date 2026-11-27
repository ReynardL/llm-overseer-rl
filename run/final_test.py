"""Final test: the whole test split once per checkpoint, greedy, on the trainer's in-process engine (the
val engine), through the orchestrator's launch() with its gates and watch.sh. The orchestrator itself never
reads test, so this lives here.

  python3 run/final_test.py <run_dir> --spec main@pick main@e0 [--go] [--fake]

spec: <arm>@pick (the arm's final pick) or <arm>@<eval id>. The arm must be done. Every spec resolves before any launch;
a finished test (done.json) is never rerun. Results: run_dir/test/<arm>_<eval>.json (with the paired delta vs the base
test main_e0 when that exists) and test_done events. Without --go it prints the commands and launches nothing.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import orchestrator as o  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--spec", nargs="+", required=True)
ap.add_argument("--tasks", default=str(o.TEST_TASKS))
ap.add_argument("--go", action="store_true")
ap.add_argument("--fake", action="store_true")  # CPU check: fake_train.py in place of the GPU launch
a = ap.parse_args()
o.FAKE["on"] = a.fake
run = o.Run(a.run_dir)
assert not run.pause.exists(), f"{run.pause} exists: the trainer would pause at once; remove it first"

todo = []
for spec in a.spec:
    arm, which = spec.split("@")
    A = run.t["arms"][arm]
    assert A["stage"]["name"] == "done", f"{spec}: arm {arm} is not done"
    eid = o.final_pick(A)["id"] if which == "pick" else which
    e = A["evals"][eid]
    argv = o.trainer_argv(run, arm, e["node"], e["gen"], "eval")
    if e["gen"]:
        assert o.complete(Path(e["ckpt"])), f"{spec}: {e['ckpt']} is not a complete checkpoint"
        argv[argv.index("--init-adapter") + 1] = e["ckpt"]  # the eval's own checkpoint (A2 seed 1's root evals live in main's dirs)
    else:
        assert "--init-adapter" not in argv
    name = f"{arm}_{eid}"
    out = run.d / "test" / name
    argv[argv.index("--val-tasks") + 1], argv[argv.index("--val-out") + 1] = a.tasks, str(out)
    # its own --out: the trainer logs episodes there (trajectories.jsonl), and A2 seed 1 copies n0's training logs
    argv[argv.index("--out") + 1] = str(run.d / "test" / f"{name}.out")
    todo.append((spec, name, arm, e, out, argv + ["--allow-test"]))

todo.sort(key=lambda x: x[1] != "main_e0")  # the base first, so the others get their paired delta vs it
for spec, name, arm, e, out, argv in todo:
    if (out / "done.json").exists():
        print("TEST already done", spec, out, flush=True)
        continue
    if not a.go:
        print("DRY test", spec, " ".join(argv), flush=True)
        continue
    rc = o.launch(run, f"test_{name}", argv, "test", lambda: True)
    if rc or not (out / "done.json").exists():
        run.event("test_failed", spec=spec, exit=rc)
        sys.exit(abs(rc or 1))
    v = o.val_summary(out / "traj.jsonl")
    base = run.d / "test" / "main_e0" / "traj.jsonl"
    vs = o.paired(base, out / "traj.jsonl") if name != "main_e0" and (base.parent / "done.json").exists() else None
    o.write_json(run.d / "test" / f"{name}.json", {"spec": spec, "arm": arm, "eval": e["id"], "node": e["node"], "gen": e["gen"],
                                                 "ckpt": e["ckpt"], "val_pass_rate": e["val"]["pass_rate"], **v, "vs_base": vs})
    run.event("test_done", spec=spec, eval=e["id"], node=e["node"], gen=e["gen"], pass_rate=v["pass_rate"], se=v["se"], n=v["n"],
              vs_base=vs and {"delta": vs["delta"], "se": vs["se"]})
    print("TEST", spec, json.dumps({"eval": e["id"], "pass_rate": v["pass_rate"], "se": v["se"], "n": v["n"],
                                    "vs_base": vs and [vs["delta"], vs["se"]]}), flush=True)
