"""CPU tests of run/orchestrator.py with fake_train.py and fake_overseer.py (WSL system python3, no GPU).

  python3 test_orch.py [--out ~/llm-overseer-rl/runs/test_orch]

Scenarios: a full chain to done with exact budget accounting; scripted CONTINUE, NEW_PHASE and
REVERT (onto an abandoned branch's eval point, and to the base) with the sampler snapshot each new phase starts from;
invalid answers (retry with the error text, then the CONTINUE fallback); no answer (retried, then exit 4 with the
decision pending and no fallback); pause mid-segment and mid-eval; RAM kills under supervise (generations redone, counted
once); A2 seeds 1 and 2 with the frozen config, the gate and A2_HOLD; the final pick on a hand-made tree; the prompt at
30 journal entries (no task id, under the token limit, compaction); crash recovery (killed mid-call, eval done but not
recorded, failed launch, partial checkpoint); dry runs; train_summary; the gates; the outage waits.
Also covered: a failed call check or a second fallback in a row exits 4 and never falls back; the
A2 warning when the overseer arm never left n0; a nonzero exit after a complete segment goes on; a rollout-logps
fallback stops finish_segment on every rerun; the compiled trainer's sha256 in launch events; REVERT cannot name the tip.
"""
import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "run"
sys.path.insert(0, str(RUN))
import knobs  # noqa: E402
import sampler  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="~/llm-overseer-rl/runs/test_orch")
a = ap.parse_args()
ROOT = Path(os.path.expanduser(a.out))
shutil.rmtree(ROOT, ignore_errors=True)
ROOT.mkdir(parents=True)
val = ROOT / "val.txt"
val.write_text("".join(list(open(HERE.parent / "data" / "val_tasks.txt"))[:160]))
(ROOT / "easy.txt").write_text("".join(list(open(HERE.parent / "data" / "train_pool.txt"))[:40]))
(ROOT / "hard.txt").write_text("".join(list(open(HERE.parent / "data" / "hard_pool.txt"))[:24]))
POOLS = {p: (ROOT / f"{p}.txt").read_text().split() for p in ("easy", "hard")}
VAL_IDS = val.read_text().split()
# 3 generations per segment, 6 segments: evals e0..e6, decisions after e1..e5
CFG = {"run_id": "test", "model": "fake", "tasks_dir": "/nonexistent", "pool": str(ROOT / "easy.txt"), "hard_pool": str(ROOT / "hard.txt"),
       "val_tasks": str(val), "budget_rollouts": 576, "eval_every_rollouts": 96, "save_gens": 2, "overseer_timeout_s": 3,
       "overseer_retry_waits_s": [1, 1, 1]}
fails = []


def check(name, cond, detail=""):
    print(("ok  " if cond else "FAIL"), name, "" if cond else str(detail)[:1500], flush=True)
    if not cond:
        fails.append(name)


def orch(d, *args, scenario=None, overseer=None):
    env = {**os.environ, "OVERSEER_CMD": f"{sys.executable} {HERE / 'fake_overseer.py'}", "OVERSEER_TMP": str(ROOT / "otmp")}
    for k, v, f in (("FAKE_SCENARIO", scenario, "scenario"), ("FAKE_OVERSEER", overseer, "overseer")):
        env.pop(k, None)
        if v is not None:
            p = d.parent / f"{d.name}.{f}.json"
            p.write_text(json.dumps(v))
            env[k] = str(p)
    r = subprocess.run([sys.executable, str(RUN / "orchestrator.py"), *args[:1], str(d), *args[1:]], env=env,
                       capture_output=True, text=True, timeout=900)
    (d.parent / f"{d.name}.out").open("a").write(r.stdout + r.stderr)
    return r


def new(name, **cfg):
    d = ROOT / name
    c = ROOT / f"{name}.cfg.json"
    c.write_text(json.dumps({**CFG, **cfg}))
    r = orch(d, "init", "--config", str(c))
    assert r.returncode == 0, r.stderr
    return d


def tree(d):
    return json.loads((d / "tree.json").read_text())


def main(d):
    return tree(d)["arms"]["main"]


def events(d):
    return [json.loads(ln) for ln in open(d / "events.log")] if (d / "events.log").exists() else []


def ev(d, kind):
    return [e for e in events(d) if e["event"] == kind]


def launches(d, node):
    f = (d / "nodes" / node if not node.startswith("seed") else d / "baseline" / node) / "checkpoints" / "fake_launches.jsonl"
    return [json.loads(ln) for ln in open(f)] if f.exists() else []


def arg(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def evs(A):
    return [A["evals"][e] for e in A["eval_order"]]


def results(A):
    """What an arm measured, eval by eval: where, the val rewards, the training summary (same-result comparisons)."""
    return [(e["gen"], e["lineage_rollouts"], e["val"]["rewards_sha"], e["train"]) for e in evs(A)]


def vals(A):
    return [(e["gen"], e["val"]["rewards_sha"]) for e in evs(A)]


def prompt(d, call):
    return (d / "overseer" / f"{call:03d}_decide.prompt.txt").read_text()


GO = ("--go", "--fake")
PHASE = {"num_generations": 8, "rollouts_per_update": 32, "hard_share": 0.5, "reset_optimizer": False}

# ---------------- full chain to done, every decision CONTINUE
d = new("happy")
r = orch(d, "run", *GO)
t = tree(d)
M = t["arms"]["main"]
n0 = M["nodes"]["n0"]
E = evs(M)
check("happy: finishes", r.returncode == 0 and M["stage"]["name"] == "done" and "ARM DONE main" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:])
check("happy: rollouts_used = budget = 32 x generations trained", M["rollouts_used"] == M["budget"] == 576 == 32 * n0["gens"]
      and M["order"] == ["n0"] and n0["status"] == "done", (M["rollouts_used"], n0["gens"]))
check("happy: base eval, then an eval point every 3 generations", [(e["id"], e["gen"], e["lineage_rollouts"], e["prev"]) for e in E]
      == [("e0", 0, 0, None)] + [(f"e{i}", 3 * i, 96 * i, f"e{i - 1}") for i in range(1, 7)], [(e["id"], e["gen"], e["prev"]) for e in E])
check("happy: a decision after every eval except the base and the last", [(j["index"], j["eval"], j["decision"], j["fallback"]) for j in t["journal"]]
      == [(i, f"e{i}", "CONTINUE", False) for i in range(1, 6)] and [e["role"] for e in ev(d, "overseer")] == ["decide"] * 5
      and [e["after_decision"] for e in E] == [None, None, 1, 2, 3, 4, 5], [(j["index"], j["eval"]) for j in t["journal"]])
check("happy: journal.jsonl mirrors the tree's journal", [json.loads(ln) for ln in open(d / "journal.jsonl")] == t["journal"]
      and all(set(j) >= {"index", "eval", "rollouts_used", "decision", "target", "config", "changed", "reason", "prediction", "change_mind",
                         "grade_previous", "fallback", "overseer"} for j in t["journal"])
      and t["journal"][0]["overseer"]["models"] == ["claude-fable-5-2"] and t["journal"][0]["overseer"]["effort"] == "xhigh"
      and t["journal"][0]["overseer"]["input_tokens"] > 0 and t["journal"][0]["rollouts_used"] == 96, t["journal"][0])
od = d / "overseer"
check("happy: request, prompt, schema, raw and response for every call",
      all((od / f"{i:03d}_decide.{x}").exists() for i in range(1, 6)
          for x in ("request.json", "prompt.txt", "schema.json", "raw.json", "raw.stream.jsonl", "response.json")))
req = json.loads((od / "001_decide.request.json").read_text())
check("happy: the request pins the model, the effort and the prompt version", req["model"] == "claude-fable-5-2" and req["effort"] == "xhigh"
      and req["prompt_version"].startswith("p5-") and req["schema"]["properties"]["target"]["enum"] == ["e0"]
      and json.loads((od / "005_decide.request.json").read_text())["schema"]["properties"]["target"]["enum"] == ["e0", "e1", "e2", "e3", "e4"],
      (req["prompt_version"], req["schema"]["properties"]["target"]))
L = launches(d, "n0")
check("happy: one base eval launch, then one launch per segment; the node resumes its own checkpoints", len(L) == 7 and "--eval-only" in L[0]
      and [arg(x, "--stop-at-gen") for x in L[1:]] == ["3", "6", "9", "12", "15", "18"] and "--resume" not in L[1] and "--init-adapter" not in L[1]
      and all("--resume" in x for x in L[2:]) and all(arg(x, "--max-gens") == "18" and arg(x, "--save-gens") == "2" for x in L), L[1])
ck = sorted(int(p.name.split("-")[1]) for p in (d / "nodes/n0/checkpoints").glob("checkpoint-*"))
check("happy: eval-point checkpoints are kept, the others pruned", ck == [6, 12, 18, 24, 30, 36]
      and [e["ckpt"] for e in E] == [None] + [str(d / f"nodes/n0/checkpoints/checkpoint-{6 * i}") for i in range(1, 7)], ck)
s1 = E[1]["train"]
check("happy: training summary of a segment", s1["rollouts"] == 96 and E[0]["train"] is None
      and all(0 <= s1[k] <= 1 for k in ("pass", "partial", "no_submit", "wrong", "load_error", "trunc", "zero_groups"))
      and s1["partial"] <= s1["wrong"] and s1["trunc"] == round(1 / 32, 4) and s1["tool_calls"] > 1 and s1["rollouts_by_pool"] == {"easy": 96, "hard": 0}
      and s1["pass_by_pool"]["hard"] is None and s1["updates"]["first"] == {"loss": 0.01, "grad_norm": 0.1, "clip": 0.0, "entropy": None}
      and s1["updates"]["second"] == {"loss": 0.02, "grad_norm": 0.2, "clip": 0.05, "entropy": None} and s1["lr_end"] == 2e-5
      and s1["steps_logged"] == 6 and s1["peak_reserved_GiB"] == 7.4 and s1["rollout_logps_fallbacks"] == 0, s1)
st0 = sampler.new("test", "root")
check("happy: the base eval keeps the untouched sampler state; each segment's rows come from the previous eval's snapshot",
      E[0]["sampler"] == st0 and n0["rows"][:24] == sampler.assign(copy.deepcopy(st0), POOLS, 8, 0.0, 3)
      and all(n0["rows"][24 * i:24 * i + 24] == sampler.assign(copy.deepcopy(E[i]["sampler"]), POOLS, 8, 0.0, 3) for i in range(1, 6))
      and len(n0["rows"]) == 144 and E[6]["sampler"]["rollouts"] == 576)
rested = [(i, x) for i in range(1, 6) for x in E[i]["sampler"]["rest_until"]]
check("happy: the zero-signal rule rests tasks at segment boundaries and the next segment skips them", rested
      and all(x not in n0["rows"][24 * i:24 * i + 24] for i, x in rested) and E[6]["pools"]["easy"]["rested"] > 0
      and E[6]["pools"]["easy"]["active"] + E[6]["pools"]["easy"]["rested"] == 40, [e["pools"] for e in E])
check("happy: paired deltas and the trailing mean", E[0]["vs_base"] is None and E[0]["trail3"] is None and E[1]["trail3"] is None
      and E[3]["trail3"] == round(sum(e["val"]["passed"] for e in E[1:4]) / sum(e["val"]["n"] for e in E[1:4]), 6) and E[3]["vs_prev"]["delta"] == round(
          E[3]["val"]["pass_rate"] - E[2]["val"]["pass_rate"], 4) and E[3]["vs_base"]["n"] == 160 and E[3]["vs_prev"]["se"] >= 0, E[3]["vs_prev"])
p1 = prompt(d, 1)
check("happy: first prompt", "This is decision 1 of 5" in p1 and "Used 96, remaining 480. Evals remaining: 5" in p1 and "Journal: empty" in p1
      and "e1 (n0, generations 0-3, 8 prompts x 4)" in p1 and "easy (40 tasks" in p1 and "hard (24 tasks" in p1 and "160 val tasks" in p1
      and "\ne1 | n0 | e0 | 96 | " in p1 and "rejected" not in p1, p1[-3000:])
p5 = prompt(d, 5)
check("happy: later prompts carry the journal, each entry with the val result that followed", all(f"#{i} at e{i} ({96 * i} rollouts used): CONTINUE" in p5
      and f"then e{i + 1}: val " in p5 for i in range(1, 5)) and "#5 at" not in p5 and "This is decision 5 of 5" in p5, p5[-2500:])
st = json.loads(orch(d, "status").stdout)
best = max(E[2:], key=lambda e: (e["trail3"], -int(e["id"][1:])))
check("happy: status prints the final pick (highest trail-3, ties to the earliest) and the fallback count",
      st["main"]["final_pick"]["id"] == best["id"] and st["main"]["final_pick"]["trail3"] == best["trail3"] and st["fallbacks"] == 0
      and st["decisions"] == 5 and st["main"]["final_pick"]["ckpt"] == best["ckpt"] and ev(d, "arm_done")[0]["final_pick"]["id"] == best["id"], st)
HAPPY, HAPPY_VALS = results(M), vals(M)
r = orch(d, "run", *GO)
check("happy: a rerun of a finished arm does nothing", r.returncode == 0 and len(launches(d, "n0")) == 7 and len(ev(d, "overseer")) == 5)
SHA = hashlib.sha256((HERE.parent / "unsloth_compiled_cache/UnslothGRPOTrainer.py").read_bytes()).hexdigest()
le = ev(d, "launch_end")
check("launch events: each records the compiled GRPO trainer's sha256; status lists the one value, no warning", len(le) == 7
      and all(e["trainer_sha256"] == SHA for e in le) and st["trainer_sha256"] == [SHA] and st["warnings"] == [] and st["exits_after_segment"] == 0,
      (SHA, [e.get("trainer_sha256") for e in le], st.get("warnings")))
dc = ROOT / "sha_changed"
shutil.copytree(d, dc)
with open(dc / "events.log", "a") as f:
    f.write(json.dumps({"time": "2026-11-13 00:00:00", "event": "launch", "name": "a2s2_g3_train_1", "trainer_sha256": "0" * 64}) + "\n")
st2 = json.loads(orch(dc, "status").stdout)
check("launch events: status warns when a run's launches saw two values of that sha256", st2["trainer_sha256"] == sorted(["0" * 64, SHA])
      and len(st2["warnings"]) == 1 and "2 sha256 values" in st2["warnings"][0], st2.get("warnings"))
r = orch(d, "a2", "--seed", "1", *GO)
w = ev(d, "a2_warning")
check("a2 warning: an overseer arm that never left n0 is flagged when A2 starts (seed 1 is then a copy of it)", r.returncode == 0
      and "A2 WARNING: the overseer arm never left n0" in r.stdout and len(w) == 1 and w[0]["decisions"] == 5 and w[0]["fallbacks"] == 0
      and all(e.get("from_root") for e in evs(tree(d)["arms"]["a2s1"])) and not launches(d, "seed1"), (r.returncode, w, r.stdout[-600:] + r.stderr[-600:]))
st3 = json.loads(orch(d, "status").stdout)
check("a2 warning: status lists it among its warnings", any("never left n0" in x for x in st3["warnings"]), st3.get("warnings"))

# ---------------- CONTINUE, NEW_PHASE, REVERT (to an earlier point, onto the abandoned branch, to the base)
ds = new("script")
SCRIPT = {"2": {"decision": "NEW_PHASE", "config": PHASE}, "3": {"decision": "REVERT", "target": "e1", "config": {"learning_rate": 1e-5}},
          "4": {"decision": "REVERT", "target": "e3", "config": PHASE}, "5": {"decision": "REVERT", "target": "e0", "config": {"reset_optimizer": False}}}
r = orch(ds, "run", *GO, overseer=SCRIPT)
t = tree(ds)
M = t["arms"]["main"]
N, E, J = M["nodes"], evs(M), t["journal"]
check("script: finishes with the budget spent across five phases", r.returncode == 0 and M["stage"]["name"] == "done" and M["order"] == ["n0", "n1", "n2", "n3", "n4"]
      and M["rollouts_used"] == 576 == 32 * sum(n["gens"] for n in N.values()) and [N[x]["gens"] for x in M["order"]] == [6, 3, 3, 3, 3],
      (r.returncode, M["rollouts_used"], r.stdout[-1500:] + r.stderr[-1500:]))
check("script: each phase starts from the decided eval point; left phases are marked", [N[x]["from"] for x in M["order"]] == [None, "e2", "e1", "e3", "e0"]
      and [N[x]["status"] for x in M["order"]] == ["left"] * 4 + ["done"] and [(j["decision"], j["target"], j["node"]) for j in J]
      == [("CONTINUE", None, "n0"), ("NEW_PHASE", None, "n1"), ("REVERT", "e1", "n2"), ("REVERT", "e3", "n3"), ("REVERT", "e0", "n4")],
      [(j["decision"], j["target"], j["node"]) for j in J])
check("script: eval points keep their path and lineage rollouts", [(e["node"], e["gen"], e["prev"], e["lineage_rollouts"]) for e in E]
      == [("n0", 0, None, 0), ("n0", 3, "e0", 96), ("n0", 6, "e1", 192), ("n1", 3, "e2", 288), ("n2", 3, "e1", 192), ("n3", 3, "e3", 384),
          ("n4", 3, "e0", 96)], [(e["node"], e["gen"], e["prev"], e["lineage_rollouts"]) for e in E])
c12, c6, c3 = (str(ds / f"nodes/{n}/checkpoints/checkpoint-{s}") for n, s in (("n0", 12), ("n0", 6), ("n1", 3)))
F = {n: launches(ds, n)[0] for n in M["order"][1:]}
check("script: NEW_PHASE keeping the optimizer starts from the tip's adapter and optimizer", arg(F["n1"], "--init-adapter") == c12
      and arg(F["n1"], "--init-optimizer") == c12 and json.loads((ds / "nodes/n1/checkpoints/init_optimizer_check.json").read_text())["optimizer_steps"] == 12, F["n1"])
check("script: REVERT with a fresh optimizer starts from the target's adapter only", arg(F["n2"], "--init-adapter") == c6 and "--init-optimizer" not in F["n2"], F["n2"])
check("script: REVERT onto the abandoned branch's eval point", arg(F["n3"], "--init-adapter") == c3 and arg(F["n3"], "--init-optimizer") == c3
      and json.loads((ds / "nodes/n3/checkpoints/init_optimizer_check.json").read_text())["optimizer_steps"] == 15, F["n3"])
check("script: REVERT to the base starts fresh, reset_optimizer clamped to true", "--init-adapter" not in F["n4"] and "--resume" not in F["n4"]
      and N["n4"]["config"]["reset_optimizer"] is True and [c["knob"] for c in J[4]["clamps"]] == ["reset_optimizer"]
      and [e["knob"] for e in ev(ds, "clamp")] == ["reset_optimizer"] and N["n4"]["config"] == knobs.ROOT, (F["n4"], J[4]["clamps"]))
check("script: every eval point's checkpoint is kept", all(Path(e["ckpt"], "trainer_state.json").exists() for e in E[1:])
      and all(Path(e["traj"]).exists() for e in E))
check("script: a new phase's rows come from its start point's sampler snapshot",
      N["n1"]["rows"] == sampler.assign(copy.deepcopy(E[2]["sampler"]), POOLS, 4, 0.5, 3)
      and N["n2"]["rows"] == sampler.assign(copy.deepcopy(E[1]["sampler"]), POOLS, 8, 0.0, 3) == N["n0"]["rows"][24:48]
      and N["n3"]["rows"] == sampler.assign(copy.deepcopy(E[3]["sampler"]), POOLS, 4, 0.5, 3)
      and N["n4"]["rows"] == sampler.assign(sampler.new("test", "root"), POOLS, 8, 0.0, 3) == N["n0"]["rows"][:24]
      and E[4]["sampler"]["pools"]["easy"] == E[2]["sampler"]["pools"]["easy"] and E[5]["sampler"]["rollouts"] == 384)
hard = set(POOLS["hard"])
check("script: hard_share 0.5 puts 2 of 4 prompts per generation in the hard pool, counted per pool",
      [x in hard for x in N["n1"]["rows"]] == [True, True, False, False] * 3 and E[3]["train"]["rollouts_by_pool"] == {"easy": 48, "hard": 48}
      and E[3]["train"]["updates"]["second"] is None and E[3]["train"]["updates"]["first"]["loss"] == 0.01 and E[3]["pools"]["hard"]["active"] <= 24,
      (N["n1"]["rows"], E[3]["train"]))
check("script: the journal records the changed knobs (reset_optimizer apart)", J[1]["changed"] == {"num_generations": [4, 8], "rollouts_per_update": [16, 32],
      "hard_share": [0.0, 0.5]} and set(J[2]["changed"]) == {"num_generations", "rollouts_per_update", "hard_share", "learning_rate"}
      and J[2]["changed"]["learning_rate"] == [2e-5, 1e-5] and J[0]["changed"] == {} and J[1]["config"]["reset_optimizer"] is False, (J[1]["changed"], J[2]["changed"]))
check("script: a revert to the base redoes the root's first segment (same val rewards as e1); trail-3 follows the path",
      E[6]["val"]["rewards_sha"] == E[1]["val"]["rewards_sha"] and E[6]["trail3"] is None
      and E[4]["trail3"] == round((E[0]["val"]["pass_rate"] + E[1]["val"]["pass_rate"] + E[4]["val"]["pass_rate"]) / 3, 6)
      and E[5]["trail3"] == round((E[2]["val"]["pass_rate"] + E[3]["val"]["pass_rate"] + E[5]["val"]["pass_rate"]) / 3, 6))
p5 = prompt(ds, 5)
check("script: the prompt shows phases, the revert and the kept optimizer", "#3 at e3 (288 rollouts used): REVERT to e1; config changes:" in p5
      and "#2 at e2 (192 rollouts used): NEW_PHASE; config changes: num_generations 4 -> 8, rollouts_per_update 16 -> 32, hard_share 0 -> 0.5; reset_optimizer false" in p5
      and "\nn2 | e1 | learning_rate=1e-05 | 3 | left" in p5 and "\ne4 | n2 | e1 | 192 | " in p5 and "then e5: val " in p5
      and "e3 (n1, generations 0-3, 4 prompts x 8)" in p5, p5[-4000:])
check("script: no prompt holds a task id", not any(x in prompt(ds, i) for i in range(1, 6) for x in VAL_IDS + POOLS["easy"] + POOLS["hard"]))

# ---------------- A2: tampering, hold, seed 1 (continues the root node), seed 2 (from the base, RAM kill under supervise)
dt = ROOT / "a2_tamper"
shutil.copytree(ds, dt)
rc_file = dt / "root_config.json"
orig = rc_file.read_text()
rc_file.write_text(orig.replace("2e-05", "3e-05"))
r = orch(dt, "a2", "--seed", "2", *GO)
check("a2: a changed root_config.json is refused", r.returncode == 1 and "root_config.json changed since init" in r.stderr and "a2s2" not in tree(dt)["arms"], r.stderr[-500:])
rc_file.write_text(orig)
tt = tree(dt)
tt["arms"]["main"]["nodes"]["n0"]["config"]["temperature"] = 0.9
(dt / "tree.json").write_text(json.dumps(tt))
r = orch(dt, "a2", "--seed", "2", *GO)
check("a2: a root node that did not run the frozen config is refused", r.returncode == 1 and "did not run the frozen root config" in r.stderr
      and "a2s2" not in tree(dt)["arms"], r.stderr[-500:])
(ds / "A2_HOLD").write_text("2")
r = orch(ds, "supervise", "--fake", "--seed", "2")
check("A2_HOLD: a held seed exits 9 under supervise, logged, no arm, no launch", r.returncode == 9 and len(ev(ds, "a2_held")) == 1
      and "a2s2" not in tree(ds)["arms"] and not ev(ds, "auto_resume") and not launches(ds, "seed2"), (r.returncode, r.stdout[-400:]))
(ds / "A2_HOLD").unlink()
calls_before = len(ev(ds, "overseer"))
r = orch(ds, "a2", "--seed", "1", *GO)
t = tree(ds)
A = t["arms"]["a2s1"]
a1 = A["nodes"]["a2s1"]
L = launches(ds, "seed1")
g = ev(ds, "a2_gate")
check("a2 seed 1: finishes with the overseer arm's rollouts, the gate logged", r.returncode == 0 and A["stage"]["name"] == "done" and A["budget"] == 576
      == A["rollouts_used"] and a1["gens"] == 18 and g and g[-1]["passed"] and g[-1]["gap"] >= 0.05, (r.returncode, r.stdout[-800:] + r.stderr[-800:]))
check("a2 seed 1: continues the root node from its last eval point with the frozen root config", a1["config"] == knobs.ROOT == t["root_config"]
      and [(e["id"], e.get("from_root", False)) for e in evs(A)] == [("e0", True), ("e1", True), ("e2", True)] + [(f"e{i}", False) for i in range(3, 7)]
      and "--resume" in L[0] and arg(L[0], "--stop-at-gen") == "9" and arg(L[0], "--seed") == "0" and len(L) == 4
      and (ds / "baseline/seed1/checkpoints/checkpoint-12/trainer_state.json").exists() and a1["stream"] == "root"
      and a1["rows"][:48] == N["n0"]["rows"] and ev(ds, "a2_seed1_from_root")[0]["gen"] == 6, (L[0], [(e["id"], e["node"]) for e in evs(A)]))
check("a2 seed 1: the same run as the root config trained straight through (the all-CONTINUE chain)", results(A) == HAPPY, vals(A))
check("a2 seed 1: no overseer call, and no warning when the overseer arm left n0", len(ev(ds, "overseer")) == calls_before
      and all(e["after_decision"] is None for e in evs(A)) and not ev(ds, "a2_warning") and "A2 WARNING" not in r.stdout)
r = orch(ds, "supervise", "--fake", "--seed", "2", scenario={"ram_kill": ["seed2:5"]})
t = tree(ds)
A = t["arms"]["a2s2"]
a2 = A["nodes"]["a2s2"]
L = launches(ds, "seed2")
check("a2 seed 2: from the base with its own seed and stream, reusing the base eval; a RAM kill resumes once", r.returncode == 0 and A["stage"]["name"] == "done"
      and A["rollouts_used"] == 576 and a2["gens"] == 18 and a2["config"] == knobs.ROOT and a2["stream"] == "a2s2" and arg(L[0], "--seed") == "2"
      and "--resume" not in L[0] and "--init-adapter" not in L[0] and "--eval-only" not in L[0] and evs(A)[0].get("from_root")
      and len(evs(A)) == 7 and [e["reason"] for e in ev(ds, "auto_resume")] == ["ram_kill"]
      and json.loads((ds / "baseline/seed2/stream.json").read_text())["name"] == "a2s2", (r.returncode, L[0], r.stdout[-800:] + r.stderr[-800:]))
check("a2 seed 2: its own task rows (another stream), the same eval cadence, no overseer call", a2["rows"][:24] != N["n0"]["rows"][:24]
      and a2["rows"][:24] == sampler.assign(sampler.new("test", "a2s2"), POOLS, 8, 0.0, 3) and vals(A) == HAPPY_VALS
      and len(ev(ds, "overseer")) == calls_before and len(open(str(ds.parent / "script.overseer.json") + ".calls").readlines()) == 5)
st = json.loads(orch(ds, "status").stdout)
check("a2: status prints each arm's final pick with the same function", all(st[x]["final_pick"]["id"] in tree(ds)["arms"][x]["eval_order"]
      for x in ("main", "a2s1", "a2s2")) and st["a2s1"]["final_pick"]["trail3"] == max(e["trail3"] for e in evs(t["arms"]["a2s1"])[2:]), st)
le = ev(ds, "launch_end")
check("a2: the A2 launches record the trainer sha256 too; one value over the whole run", st["trainer_sha256"] == [SHA] and st["warnings"] == []
      and {e["name"].split("_")[0] for e in le} == {"n0", "n1", "n2", "n3", "n4", "a2s1", "a2s2"} and all(e["trainer_sha256"] == SHA for e in le),
      (st["trainer_sha256"], sorted({e["name"].split("_")[0] for e in le})))
dg = new("a2_gate")
orch(dg, "run", *GO, scenario={"flat": True})
r = orch(dg, "a2", "--seed", "1", *GO, scenario={"flat": True})
g = ev(dg, "a2_gate")
check("a2 gate: refused when best val - base val < 0.05 (exit 7), logged, no arm, no launch", r.returncode == 7 and len(g) == 1 and not g[0]["passed"]
      and g[0]["gap"] == 0.0 and g[0]["min_gap"] == 0.05 and list(tree(dg)["arms"]) == ["main"] and not launches(dg, "seed1"), (r.returncode, g))
check("a2 gate: only the smoke config lowers it", "a2_min_gap" not in json.loads((HERE.parent / "configs/root.json").read_text())
      and json.loads((HERE.parent / "configs/smoke.json").read_text())["a2_min_gap"] < 0)
de = new("a2_early")
orch(de, "run", *GO, "--max-launches", "1")
r = orch(de, "a2", "--seed", "1", *GO)
check("a2: refused before the overseer arm is done", r.returncode == 1 and "A2 runs after the overseer arm finishes" in r.stderr, r.stderr[-300:])

import orchestrator as O  # noqa: E402
import prompts as PR  # noqa: E402

# ---------------- final pick on a hand-made tree
def hand(spec):
    A = {"evals": {i: {"id": i, "node": "n", "gen": 0, "prev": p, "ckpt": i, "val": {"pass_rate": v, "passed": round(v * 10), "n": 10}} for i, p, v in spec}, "eval_order": [s[0] for s in spec]}
    for i in A["eval_order"]:
        A["evals"][i]["trail3"] = O.trail3(A, i)
    return A


A = hand([("e0", None, 0.2), ("e1", "e0", 0.3), ("e2", "e1", 0.4), ("e3", "e2", 0.2), ("e5", "e0", 0.9)])
check("final pick: trail-3 over the path, ties to the earliest, paths under 3 evals never picked", O.final_pick(A)["id"] == "e2"
      and [A["evals"][i]["trail3"] for i in A["eval_order"]] == [None, None, 0.3, 0.3, None] and O.path(A, "e3") == ["e0", "e1", "e2", "e3"],
      [A["evals"][i]["trail3"] for i in A["eval_order"]])
A = hand([("e0", None, 0.2), ("e1", "e0", 0.3), ("e2", "e1", 0.4), ("e3", "e2", 0.2), ("e4", "e1", 0.5), ("e5", "e4", 0.1)])
check("final pick: a revert's path skips the abandoned evals", O.final_pick(A)["id"] == "e4" and A["evals"]["e4"]["trail3"] == round(1.0 / 3, 6)
      and A["evals"]["e5"]["trail3"] == 0.3 and O.final_pick(hand([("e0", None, 0.2), ("e1", "e0", 0.9)])) is None, O.final_pick(A))

# ---------------- invalid answers: one retry with the error text, then the CONTINUE fallback
d = new("retry")
r = orch(d, "run", *GO, "--max-launches", "2", overseer={"1": "badtarget"})
t = tree(d)
check("invalid answer: retried once with the error text, the retry's answer is used", r.returncode == 0 and len(t["journal"]) == 1
      and not t["journal"][0]["fallback"] and t["journal"][0]["overseer"]["call"] == 2 and len(ev(d, "decision_invalid")) == 1 and not ev(d, "fallback")
      and "Your previous answer was rejected: REVERT target 'e99' is not one of the kept eval points ['e0']" in prompt(d, 2)
      and "rejected" not in prompt(d, 1), (t["journal"], ev(d, "decision_invalid")))
d = new("fallback")
r = orch(d, "run", *GO, "--max-launches", "2", overseer={"1": "invalid", "2": "badtarget"})
t = tree(d)
j = t["journal"][0]
check("two invalid answers: CONTINUE fallback, journaled with fallback true, the same phase trains on", r.returncode == 0 and len(t["journal"]) == 1 and j["fallback"]
      and j["decision"] == "CONTINUE" and j["node"] == "n0" and j["reason"] == "" and len(ev(d, "decision_invalid")) == 2 and len(ev(d, "fallback")) == 1
      and t["arms"]["main"]["stage"] == {"name": "train", "node": "n0", "seg_end": 6} and t["calls"] == 2
      and json.loads(orch(d, "status").stdout)["fallbacks"] == 1, (j, t["arms"]["main"]["stage"]))
orch(d, "run", *GO, "--max-launches", "1")
check("fallback: the next prompt marks the entry as a fallback", "#1 at e1 (96 rollouts used): CONTINUE [fallback: no valid answer, the run continued]" in prompt(d, 3))
d = new("revert_tip")
r = orch(d, "run", *GO, "--max-launches", "2", overseer={"1": {"decision": "REVERT", "target": "e1", "config": {}}})
t = tree(d)
check("REVERT to the tip: rejected as a content error (the tip is not a target), the retry's answer is used", r.returncode == 0 and len(t["journal"]) == 1
      and t["journal"][0]["decision"] == "CONTINUE" and not t["journal"][0]["fallback"] and t["journal"][0]["overseer"]["call"] == 2
      and [e["error"] for e in ev(d, "decision_invalid")] == ["REVERT target 'e1' is not one of the kept eval points ['e0']"]
      and t["arms"]["main"]["order"] == ["n0"] and "any id in the eval table except the latest" in prompt(d, 1), (t["journal"], ev(d, "decision_invalid")))
d = new("fallback2")
r = orch(d, "run", *GO, overseer={str(i): "invalid" for i in range(1, 5)})
t = tree(d)
check("second fallback in a row: exit 4 with the decision pending, not a second CONTINUE", r.returncode == 4 and len(t["journal"]) == 1
      and t["journal"][0]["fallback"] and t["arms"]["main"]["stage"] == {"name": "decide", "eval": "e2"} and len(ev(d, "fallback")) == 1
      and [e["why"] for e in ev(d, "decision_halt")] == ["second fallback in a row"] and len(launches(d, "n0")) == 3 and t["calls"] == 4,
      (r.returncode, t["arms"]["main"]["stage"], ev(d, "decision_halt")))
r = orch(d, "run", *GO, "--halt-before", "train", overseer={})
t = tree(d)
check("second fallback in a row: a rerun takes the decision again with a new call", r.returncode == 0 and len(t["journal"]) == 2
      and not t["journal"][1]["fallback"] and t["journal"][1]["overseer"]["call"] == 5 and t["journal"][1]["overseer"]["attempt"] == 1
      and "rejected" not in prompt(d, 5), t["journal"][1:])
d = new("fallback_apart")
r = orch(d, "run", *GO, overseer={"1": "invalid", "2": "invalid", "5": "invalid", "6": "badtarget"})
t = tree(d)
check("fallbacks that are not in a row are allowed: the run finishes, status counts them", r.returncode == 0
      and [j["fallback"] for j in t["journal"]] == [True, False, False, True, False] and not ev(d, "decision_halt")
      and t["arms"]["main"]["stage"]["name"] == "done" and json.loads(orch(d, "status").stdout)["fallbacks"] == 2, [j["fallback"] for j in t["journal"]])
for cls in ("badmodel", "nomodel", "turns", "mcp", "tokens"):
    d = new(f"inv_{cls}")
    r = orch(d, "run", *GO, overseer={"1": cls, "2": cls})
    t = tree(d)
    check(f"failed call check ({cls}): never acted on, one retry with no error text, then exit 4 with the decision pending, no fallback",
          r.returncode == 4 and not t["journal"] and not ev(d, "fallback") and t["arms"]["main"]["stage"] == {"name": "decide", "eval": "e1"}
          and len(ev(d, "decision_invalid")) == 2 and all(e["error"] for e in ev(d, "decision_invalid")) and "rejected" not in prompt(d, 2)
          and [e["why"] for e in ev(d, "decision_halt")] == ["a call failed a model, isolation or token check"] and len(launches(d, "n0")) == 2
          and t["arms"]["main"]["order"] == ["n0"], (r.returncode, t["arms"]["main"]["stage"], ev(d, "decision_invalid")))
    r = orch(d, "run", *GO, "--halt-before", "train", overseer={})
    t = tree(d)
    check(f"failed call check ({cls}): a rerun takes the decision again with a new call", r.returncode == 0 and len(t["journal"]) == 1
          and not t["journal"][0]["fallback"] and t["journal"][0]["overseer"]["call"] == 3, t["journal"])
for name, acts in (("check_then_content", {"1": "turns", "2": "invalid"}), ("content_then_check", {"1": "badtarget", "2": "mcp"})):
    d = new(name)
    r = orch(d, "run", *GO, overseer=acts)
    check(f"one failed call check and one content error ({name}): exit 4, no fallback", r.returncode == 4 and not tree(d)["journal"]
          and not ev(d, "fallback") and len(ev(d, "decision_halt")) == 1 and main(d)["stage"]["name"] == "decide", (r.returncode, ev(d, "decision_halt")))
d = new("clamp")
r = orch(d, "run", *GO, "--max-launches", "2", overseer={"1": "clamp"})
t = tree(d)
check("clamp: an out-of-range number is clamped and logged, the phase starts", r.returncode == 0 and t["arms"]["main"]["nodes"]["n1"]["config"]["learning_rate"] == 1e-4
      and t["journal"][0]["clamps"][0]["knob"] == "learning_rate" and ev(d, "clamp")[0]["from"] == 1e-3 and not t["journal"][0]["fallback"]
      and t["journal"][0]["changed"] == {"learning_rate": [2e-5, 1e-4]}, t["journal"][0])

# ---------------- no answer: retried, then exit 4 with the decision pending; never a fallback; a rerun re-sends
for act in ("hang", "error", "iserror", "claudeexit"):
    d = new(f"out_{act}")
    r = orch(d, "run", *GO, overseer={"1": act})
    t = tree(d)
    check(f"overseer unavailable ({act}): 4 attempts, exit 4, decision pending, no fallback, no response", r.returncode == 4
          and t["arms"]["main"]["stage"]["name"] == "decide" and t["arms"]["main"]["stage"]["call"] == 1 and not t["journal"] and not ev(d, "fallback")
          and len(ev(d, "overseer_retry")) == 3 and len(ev(d, "overseer_unavailable")) == 1 and not list((d / "overseer").glob("*.response.json"))
          and (d / "overseer/001_decide.attempt4.unavailable.json").exists() and len(launches(d, "n0")) == 2, (r.returncode, t["arms"]["main"]["stage"]))
    r = orch(d, "run", *GO, "--max-launches", "1", overseer={})
    t = tree(d)
    check(f"overseer unavailable ({act}): a rerun re-sends the same call and goes on", r.returncode == 0 and t["journal"][0]["overseer"]["call"] == 1
          and t["journal"][0]["overseer"]["attempt"] == 5 and not t["journal"][0]["fallback"] and len(launches(d, "n0")) == 3, t["journal"][:1])

# ---------------- pause mid-segment and mid-eval
d = new("pause_train")
r = orch(d, "run", *GO, scenario={"pause_train": ["n0:2"]})
A = main(d)
check("pause mid-segment: exit 3 at the generation boundary, nothing counted yet", r.returncode == 3 and A["stage"]["name"] == "train" and A["rollouts_used"] == 0
      and (d / "nodes/n0/checkpoints/checkpoint-4").exists() and len(A["eval_order"]) == 1, (r.returncode, A["stage"]))
r = orch(d, "run", *GO, scenario={"pause_train": ["n0:2"]})
check("pause: a run with PAUSE present does nothing", r.returncode == 3 and len(launches(d, "n0")) == 2)
(d / "PAUSE").unlink()
r = orch(d, "run", *GO, scenario={"pause_train": ["n0:2"]})
check("pause mid-segment: the resume finishes with the same results", r.returncode == 0 and "--resume" in launches(d, "n0")[2]
      and results(main(d)) == HAPPY and main(d)["rollouts_used"] == 576, r.stdout[-800:] + r.stderr[-800:])
d = new("pause_eval")
r = orch(d, "run", *GO, scenario={"pause_eval": ["n0:3"]})
A = main(d)
check("pause mid-eval: exit 3, the stage becomes eval", r.returncode == 3 and A["stage"]["name"] == "eval" and A["stage"]["seg_end"] == 3
      and not (d / "nodes/n0/val/3/done.json").exists(), (r.returncode, A["stage"]))
(d / "PAUSE").unlink()
r = orch(d, "run", *GO, scenario={"pause_eval": ["n0:3"]})
L = launches(d, "n0")
check("pause mid-eval: an eval-only launch from the segment's checkpoint, same results", r.returncode == 0 and "--eval-only" in L[2]
      and arg(L[2], "--init-adapter") == str(d / "nodes/n0/checkpoints/checkpoint-6") and results(main(d)) == HAPPY, L[2])

# ---------------- RAM kills under supervise: generations are redone and counted once
d = new("sup_two")
r = orch(d, "supervise", "--fake", scenario={"ram_kill": ["n0:3", "n0:8"]})
ar, lf = ev(d, "auto_resume"), ev(d, "launch_failed")
nrows = len(open(d / "nodes/n0/checkpoints/rows.jsonl").readlines())
check("supervise: two RAM kills each resume once; a generation without a checkpoint is redone and counted once", r.returncode == 0
      and [(e["reason"], e["count"]) for e in ar] == [("ram_kill", 1), ("ram_kill", 2)] and [(e["exit"], e.get("reason")) for e in lf] == [(-5, "ram_kill")] * 2
      and results(main(d)) == HAPPY and main(d)["rollouts_used"] == 576 and nrows == 19 * 32, (r.returncode, ar, lf, nrows, r.stdout[-600:]))
d = new("sup_cut")  # a cut events.log line before a RAM kill: supervise still counts its resumes
with open(d / "events.log", "a") as f:
    f.write('{"time": "2026-11-13 20:00:00", "ev')
r = orch(d, "supervise", "--fake", scenario={"ram_kill": ["n0:3"]})
evc = O.Run(d).events()
check("supervise: a cut events.log line is skipped; the RAM kill resumes once and the run finishes the same",
      r.returncode == 0 and [e["count"] for e in evc if e["event"] == "auto_resume"] == [1] and results(main(d)) == HAPPY,
      (r.returncode, r.stderr[-400:]))
d = new("sup_seven")
r = orch(d, "supervise", "--fake", scenario={"ram_kill": [f"n0:{i}" for i in range(1, 8)]})
check("supervise: the 7th RAM kill within 24 h stops with exit 5 and auto_resume_refused (limit 6)", r.returncode == 5
      and [e["count"] for e in ev(d, "auto_resume")] == [1, 2, 3, 4, 5, 6] and [e["count"] for e in ev(d, "auto_resume_refused")] == [6]
      and main(d)["stage"]["name"] == "train" and O.AUTO_RESUME_MAX == 6 and O.AUTO_RESUME_WINDOW_S == 86400,
      (r.returncode, ev(d, "auto_resume"), ev(d, "auto_resume_refused")))
for name, kw, code in [("sup_oom", {"scenario": {"fail": ["n0:2"]}}, 1), ("sup_outage", {"overseer": {"1": "error"}}, 4),
                       ("sup_pause", {"scenario": {"pause_train": ["n0:2"]}}, 3)]:
    d = new(name)
    r = orch(d, "supervise", "--fake", **kw)
    check(f"supervise: {name[4:]} stops with exit {code}, no auto-resume, one orchestrator run",
          r.returncode == code and not ev(d, "auto_resume") and r.stdout.count("SUPERVISE stop") == 1, (r.returncode, r.stdout[-400:]))

# ---------------- crash recovery
d = new("die")
r = orch(d, "run", *GO, overseer={"1.1": "die"})
check("killed mid-call: request on disk, no response", r.returncode == -9 and (d / "overseer/001_decide.request.json").exists()
      and not (d / "overseer/001_decide.response.json").exists(), r.returncode)
r = orch(d, "run", *GO, "--halt-before", "train", overseer={"1.1": "die"})
t = tree(d)
req = json.loads((d / "overseer/001_decide.request.json").read_text())
check("killed mid-call: the rerun re-sends the same call (attempt 2), no extra launch", r.returncode == 0 and req["attempt"] == 2
      and len(ev(d, "overseer_resend")) == 1 and t["calls"] == 1 and len(launches(d, "n0")) == 2 and len(t["journal"]) == 1, (r.stdout[-800:], req["attempt"]))
before = (d / "overseer/001_decide.response.json").read_text()
t["arms"]["main"]["stage"] = {"name": "decide", "eval": "e1", "call": 1}  # as if killed after the response was written, before acting
t["journal"] = []
(d / "tree.json").write_text(json.dumps(t))
(d / "journal.jsonl").unlink()
cnt = len(ev(d, "overseer"))
orch(d, "run", *GO, "--halt-before", "train")
t = tree(d)
check("response on disk: acted on without a new call, the decision journaled once", len(ev(d, "overseer")) == cnt and t["arms"]["main"]["stage"]["name"] == "train"
      and len(t["journal"]) == 1 and len(open(d / "journal.jsonl").readlines()) == 1 and (d / "overseer/001_decide.response.json").read_text() == before)
d = new("teardown")
r = orch(d, "run", *GO, scenario={"fail_after_eval": ["n0:3"]})
x = ev(d, "exit_after_segment")
check("nonzero exit after a complete segment (done.json written): logged with the exit code, the run goes on, same results; status counts it",
      r.returncode == 0 and len(x) == 1 and x[0]["exit"] == 1 and x[0]["stage"]["seg_end"] == 3 and not ev(d, "launch_failed")
      and not ev(d, "segment_recovered") and results(main(d)) == HAPPY and len(launches(d, "n0")) == 7
      and json.loads(orch(d, "status").stdout)["exits_after_segment"] == 1, (r.returncode, x, r.stdout[-800:] + r.stderr[-800:]))
d = new("recover")
orch(d, "run", *GO, "--max-launches", "1")
before = (d / "tree.json").read_text()  # the base eval is recorded, the first segment is pending
orch(d, "run", *GO, "--halt-before", "decide")
(d / "tree.json").write_text(before)  # as if the orchestrator was killed after the trainer wrote done.json, before its own save
check("eval done but not recorded: stage still train, nothing counted", main(d)["stage"]["name"] == "train"
      and (d / "nodes/n0/val/3/done.json").exists() and main(d)["rollouts_used"] == 0 and len(main(d)["eval_order"]) == 1, main(d)["stage"])
r = orch(d, "run", *GO)
check("eval done but not recorded: the rerun records it without a launch, same results", r.returncode == 0 and len(ev(d, "segment_recovered")) == 1
      and len([x for x in launches(d, "n0") if arg(x, "--stop-at-gen") == "3"]) == 1 and results(main(d)) == HAPPY, (r.returncode, r.stdout[-800:]))
d = new("logps_fallback")
r = orch(d, "run", *GO, scenario={"logps_fallback": ["n0:2"]})
check("rollout-logps fallback in a segment: finish_segment refuses it (exit 1), nothing recorded", r.returncode == 1
      and "rollout logprobs as the old policy: fallbacks 1" in r.stderr and main(d)["stage"]["name"] == "train" and main(d)["rollouts_used"] == 0
      and len(main(d)["eval_order"]) == 1 and (d / "nodes/n0/val/3/done.json").exists(), (r.returncode, r.stderr[-600:]))
r = orch(d, "run", *GO, scenario={"logps_fallback": ["n0:2"]})
check("rollout-logps fallback: a rerun does not get past it and launches nothing", r.returncode == 1 and "fallbacks 1" in r.stderr
      and len(launches(d, "n0")) == 2 and main(d)["rollouts_used"] == 0 and len(main(d)["eval_order"]) == 1, (r.returncode, len(launches(d, "n0"))))
d = new("fail")
r = orch(d, "run", *GO, scenario={"fail": ["n0:2"]})
check("failed launch: exit 1, launch_failed logged, no retry", r.returncode == 1 and len(ev(d, "launch_failed")) == 1 and len(launches(d, "n0")) == 2)
r = orch(d, "run", *GO, scenario={"fail": ["n0:2"]})
check("failed launch: the rerun redoes the segment to the same results", r.returncode == 0 and results(main(d)) == HAPPY)
d = new("partial")
r = orch(d, "run", *GO, scenario={"pause_train": ["n0:2"]})
part = d / "nodes/n0/checkpoints/checkpoint-6"
part.mkdir()
(part / "adapter.json").write_text("{}")  # no trainer_state.json: the save was cut
for k, s in ((8, ""), (10, '{"global_step": 1')):  # a kill inside trainer_state.json's own write: empty, or cut
    (d / f"nodes/n0/checkpoints/checkpoint-{k}").mkdir()
    (d / f"nodes/n0/checkpoints/checkpoint-{k}/trainer_state.json").write_text(s)
(d / "PAUSE").unlink()
r = orch(d, "run", *GO, scenario={"pause_train": ["n0:2"]})
pe = ev(d, "partial_checkpoint_removed")
check("partial checkpoint: removed and logged before the resume, same results", r.returncode == 0 and len(pe) == 3
      and {p["path"].rsplit("-", 1)[1] for p in pe} == {"6", "8", "10"} and results(main(d)) == HAPPY, (r.returncode, pe))
for k, s in (("unit/checkpoint-2", '{"global_step": 2}'), ("unit/checkpoint-4", ""), ("unit/checkpoint-6", '{"global_st'),
             ("unit/checkpoint-8", None), ("unit_bad/checkpoint-4", "")):
    (ROOT / "ckpts" / k).mkdir(parents=True)
    if s is not None:
        (ROOT / "ckpts" / k / "trainer_state.json").write_text(s)
check("ckpts: only a trainer_state.json that parses makes a checkpoint complete (empty, cut and missing do not; all invalid gives none)",
      O.ckpts(ROOT / "ckpts/unit") == [2] and O.ckpts(ROOT / "ckpts/unit_bad") == [], (O.ckpts(ROOT / "ckpts/unit"), O.ckpts(ROOT / "ckpts/unit_bad")))

# ---------------- dry runs, max-launches, no test command
d = new("dry")
r = orch(d, "run")
check("dry: prints the command, launches nothing, assigns nothing", r.returncode == 0 and "DRY next: eval0 n0 0" in r.stdout and "--eval-only" in r.stdout
      and not launches(d, "n0") and main(d)["stage"]["name"] == "eval0" and main(d)["nodes"]["n0"]["rows"] == [], r.stdout)
r = orch(d, "run", *GO, "--max-launches", "1")
check("max-launches 1: exactly one launch, no decision after the base eval", r.returncode == 0 and len(launches(d, "n0")) == 1
      and main(d)["stage"] == {"name": "train", "node": "n0", "seg_end": 3} and not ev(d, "overseer"), main(d)["stage"])
orch(d, "run", *GO, "--halt-before", "decide")
r = orch(d, "run")
check("dry at a decision: no overseer call", r.returncode == 0 and "DRY next: decision call at e1" in r.stdout and not ev(d, "overseer")
      and main(d)["stage"]["name"] == "decide", r.stdout)
r = orch(d, "test")
src = (RUN / "orchestrator.py").read_text()
check("no test command; the test task list is read by init's disjointness assert only", r.returncode == 2 and "invalid choice" in r.stderr
      and src.count("TEST_TASKS") == 2 and "test_tasks" not in (RUN / "prompts.py").read_text() + (RUN / "trainer.py").read_text(), r.stderr[-300:])

# ---------------- 32 segments: the last prompt holds 30 journal entries
dl = new("long", eval_every_rollouts=32, budget_rollouts=1024, save_gens=1)
LONG = {"long": True, "4": {"decision": "NEW_PHASE", "config": {"learning_rate": 1e-5}}, "9": {"decision": "REVERT", "target": "e3", "config": PHASE},
        "15": {"decision": "NEW_PHASE", "config": {"temperature": 0.9, "max_grad_norm": 0.5}}, "22": {"decision": "REVERT", "target": "e0", "config": {}},
        "27": {"decision": "NEW_PHASE", "config": {"scale_rewards": "none", "reset_optimizer": False}}}
r = orch(dl, "run", *GO, overseer=LONG)
t = tree(dl)
M = t["arms"]["main"]
p31 = prompt(dl, 31)
check("long: 32 segments, 31 decisions, budget exact", r.returncode == 0 and M["stage"]["name"] == "done" and M["rollouts_used"] == 1024
      == 32 * sum(n["gens"] for n in M["nodes"].values()) and len(t["journal"]) == 31 and len(M["eval_order"]) == 33 and len(M["order"]) == 6
      and len(open(dl / "journal.jsonl").readlines()) == 31, (r.returncode, len(t["journal"]), r.stdout[-800:] + r.stderr[-800:]))
# ~2 chars per token (measured, call 45 of the main run); 30 long fake entries pass 60K chars, so the journal is compacted
check("long: the prompt at 30 journal entries stays under the token limit and holds no task id", len(re.findall(r"(?m)^#\d+ at e\d+ ", p31)) == 30
      and len(p31) / 2 < O.MAX_INPUT_TOKENS and p31.count("  reason: ") == PR.FULL_ENTRIES and "This is decision 31 of 31" in p31
      and not any(x in p31 for x in VAL_IDS + POOLS["easy"] + POOLS["hard"]) and all(e["problems"] == [] for e in (
          json.loads(f.read_text())["isolation"] for f in (dl / "overseer").glob("*.response.json"))), (len(p31), len(re.findall(r"(?m)^#\d+ at e\d+ ", p31))))
small = PR.decision(t, None, 9000)
check("long: a journal that threatens the limit keeps the last 10 entries in full and one line for older ones",
      small.count("  reason: ") == PR.FULL_ENTRIES == 10 and len(re.findall(r"(?m)^#\d+ at e\d+ ", small)) == 31 and len(small) < len(PR.decision(t, None, 10**6))
      and "#21 at e21 (672 rollouts used): CONTINUE; then e22: val " in small, small.count("  reason: "))
check("long: the eval table has every eval point once", all(len(re.findall(rf"(?m)^{e} \| ", p31)) == 1 for e in M["eval_order"][:32]))

# ---------------- train_summary: a redone generation's last records win; rows and env log must agree
sd = ROOT / "summary_rows"
sd.mkdir()
node = {"config": dict(knobs.ROOT), "rows": [f"t{i}" for i in range(8)]}


def gen_rows(reward, bad=None):
    rows = [{"step": 0, "gen": 0, "row": j, "task_id": f"t{j // 4}", "prompt": 5, "completion": 9, "trained_tokens": 9, "masked_truncated": False,
             "advantage": 0.0, "reward": reward, "passed": float(reward == 1), "env_index": j} for j in range(32)]
    traj = [{"step": 0, "index": j, "task_id": f"t{j // 4}", "calls": [1, 2], "submitted": True, "grade": {"passed": reward == 1}, "passed": float(reward == 1),
             "reward": 0.5 if bad == j else reward} for j in range(32)]
    return "".join(json.dumps(x) + "\n" for x in rows), "".join(json.dumps(x) + "\n" for x in traj)


r0, t0 = gen_rows(0.0)
r1, t1 = gen_rows(1.0)
(sd / "rows.jsonl").write_text(r0 + '{"step": 0, "gen": 0, "ro' + "\n" + r1)  # a generation redone after a kill, with a cut line between
(sd / "trajectories.jsonl").write_text(t0 + t1)
s, visits = O.train_summary(sd, node, 0, 1, set())
check("train_summary: a redone generation's last rows replace the first; a cut line is skipped", s["rollouts"] == 32 and s["pass"] == 1.0 and s["reward"] == 1.0
      and s["zero_groups"] == 1.0 and s["wrong"] == 0.0 and s["tool_calls"] == 2 and visits == [(f"t{i}", [1.0] * 4) for i in range(8)], s)
(sd / "trajectories.jsonl").write_text(t0 + gen_rows(1.0, bad=7)[1])
for name, fn in (("a reward that differs between rows.jsonl and the env log", lambda: O.train_summary(sd, node, 0, 1, set())),
                 ("a generation missing from rows.jsonl", lambda: O.train_summary(sd, node, 0, 2, set())),
                 ("rows that are not the assigned ones", lambda: O.train_summary(sd, {**node, "rows": node["rows"][::-1]}, 0, 1, set()))):
    try:
        fn()
        check(f"train_summary refuses {name}", False)
    except AssertionError as e:
        check(f"train_summary refuses {name}", True, e)


ev_dir = new("cut_events")  # a kill inside an events.log append: status and supervise skip the cut line
O.Run(ev_dir).event("halt", before="decide")  # init logs nothing; one complete event first
with open(ev_dir / "events.log", "a") as f:
    f.write('{"time": "2026-11-13 20:00:00", "ev')
r = orch(ev_dir, "status")
evs_cut = O.Run(ev_dir).events()
check("events.log: a cut last line is skipped by Run.events (status, supervise), not fatal",
      r.returncode == 0 and [e["event"] for e in evs_cut] == ["halt"], (r.returncode, evs_cut, r.stderr[-300:]))


class Ev:  # run stub for seal_tails, which only logs
    def __init__(self):
        self.ev = []

    def event(self, k, **kw):
        self.ev.append(k)


ro = Ev()
(sd / "rows.jsonl").write_text(r0 + '{"step": 0, "gen": 0, "ro')  # a hard kill cut the last line: no newline after it
(sd / "trajectories.jsonl").write_text(t0 + '{"step": 1, "ind')
O.seal_tails(ro, sd)
O.seal_tails(ro, sd)  # a sealed file is left alone
with open(sd / "rows.jsonl", "a") as f:  # the resumed launch appends its redone generation
    f.write(r1)
with open(sd / "trajectories.jsonl", "a") as f:
    f.write(t1)
s, visits = O.train_summary(sd, node, 0, 1, set())
check("seal_tails: a cut last line is ended before the next launch appends, so the first resumed record is not lost with it",
      s["rollouts"] == 32 and s["pass"] == 1.0 and ro.ev == ["cut_line_ended"] * 2, (s.get("pass"), ro.ev))
ro = Ev()
O.seal_tails(ro, ROOT / "no_such_dir")
check("seal_tails: a node with no logs yet (the base eval) is a no-op", ro.ev == [])
(sd / "summary.jsonl").write_text(json.dumps({"global_step": 2, "peak_reserved_GiB": 6.5}) + "\n" + '{"global_step": 2, "peak_res')
s, _ = O.train_summary(sd, node, 0, 1, set())
check("train_summary: a summary.jsonl line cut by a kill after done.json (the done.json recovery reads it) is skipped",
      s["peak_reserved_GiB"] == 6.5, s.get("peak_reserved_GiB"))
(sd / "summary.jsonl").unlink()

# ---------------- final pick: computed from pass counts, so equal counts tie exactly and the earliest wins
A = {"eval_order": [f"e{i}" for i in range(7)], "evals": {}}
for i, k in enumerate([0, 1, 1, 2, 0, 0, 4]):  # e3's window (1, 1, 2) and e6's (0, 0, 4) both hold 4 passes of 1,194
    A["evals"][f"e{i}"] = {"id": f"e{i}", "node": "n0", "gen": i, "ckpt": i, "prev": f"e{i - 1}" if i else None,
                           "val": {"passed": k, "n": 398, "pass_rate": round(k / 398, 4)}}
for e in A["eval_order"]:
    A["evals"][e]["trail3"] = O.trail3(A, e)
check("final pick: equal pass counts in two windows tie (rounded pass rates would pick the later), the earliest wins",
      O.final_pick(A)["id"] == "e3" and A["evals"]["e3"]["trail3"] == A["evals"]["e6"]["trail3"], {e: A["evals"][e]["trail3"] for e in A["eval_order"]})

# ---------------- gates: which python blocks, how long a refusal waits
lines = ["101 python3 /repo/run/trainer.py --model x", "102 /usr/bin/python3 -I -S launcher.py {}",
         "103 python3 -m vllm.entrypoints.x", "104 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown",
         "105 python3 /home/user/.editor-server/x.py", "106 bash -c find / -name '*python*'", "107 python3 orchestrator.py run d",
         "108 python3 /repo/grpo_dummy.py --model x"]
check("gates: only the trainer, eval, vLLM and sandbox count as busy python", [ln.split()[0] for ln in O.our_python(lines, {107})]
      == ["101", "102", "103", "108"], O.our_python(lines, {107}))


class Clock:
    def __init__(self):
        self.t = 0.0

    def time(self):
        return self.t

    def sleep(self, x):
        self.t += x

    def strftime(self, *a):
        return "fake"


def waits(name, refused_seq):
    d = new(name)
    run, it, real, real_g = O.Run(d), iter(refused_seq), O.time, O.gates
    O.time = Clock()
    O.gates = lambda run: (lambda rf: (not rf, {"refused": rf, "windows_free_GB": "11.0"}))(next(it, refused_seq[-1]))
    try:
        rec = O.wait_idle(run)
    finally:
        O.time, O.gates = real, real_g
    return rec, ev(d, "gates_wait"), ev(d, "gates_refused"), ev(d, "gates_passed")


rec, w, rf, ps = waits("gates_ram", [["ram"]] * 200)
check("gates: RAM alone waits 2 h at 60 s per check, each refusal logged, then refuses", rec is None and len(w) == 122
      and w[-1]["waited_s"] == 7260 and all(x["refused"] == ["ram"] for x in w) and len(rf) == 1, (len(w), w[-1:], rf))
rec, w, rf, ps = waits("gates_other", [["ram", "wsl_python"]] * 200)
check("gates: another refusal waits 15 min at 30 s", rec is None and len(w) == 32 and w[-1]["waited_s"] == 930, (len(w), w[-1:]))
rec, w, rf, ps = waits("gates_recover", [["ram"]] * 30 + [[]])
check("gates: RAM recovering after 30 min passes, wait logged", rec is not None and len(w) == 30 and ps and ps[0]["waited_s"] == 1800, (len(w), ps))
d = new("ram_gate")  # in-process: a launch whose gate refuses on RAM alone for the whole 2 h -> exit 6, reason ram_gate
run, real_t, real_g = O.Run(d), O.time, O.gates
out0 = d / "nodes/n0/checkpoints"
out0.mkdir(parents=True, exist_ok=True)
(out0 / "rows.jsonl").write_text('{"cut')  # a line still being written by a trainer the gate saw, or cut by a kill
O.time, O.gates = Clock(), (lambda run: (False, {"refused": ["ram"], "windows_free_GB": "11.0"}))
try:
    rc = O.step_stage(run, True, [])
finally:
    O.time, O.gates = real_t, real_g
check("gate timeout on RAM alone: exit 6, launch_failed reason ram_gate", rc == 6 and ev(d, "launch_failed")[-1].get("reason") == "ram_gate",
      (rc, ev(d, "launch_failed")))
check("a refused launch repairs nothing: the logs are byte for byte as they were, no cut_line_ended",
      (out0 / "rows.jsonl").read_text() == '{"cut' and not ev(d, "cut_line_ended"), (out0 / "rows.jsonl").read_text())
d = new("stale")  # in-process: a trainer that outlived its orchestrator finishes the base eval during the gate wait
run, real_w = O.Run(d), O.wait_idle
out0 = d / "nodes/n0/checkpoints"
out0.mkdir(parents=True, exist_ok=True)
(out0 / "rows.jsonl").write_text('{"cut')


def survivor(run):
    (run.node_dir("n0") / "val/0").mkdir(parents=True, exist_ok=True)
    (run.node_dir("n0") / "val/0/done.json").write_text("{}")
    return {"refused": []}


O.wait_idle = survivor
try:
    rc = O.step_stage(run, True, [])
finally:
    O.wait_idle = real_w
check("stale command: the stage moved on during the gate wait, so launch_stale, nothing launched or repaired, the stage starts over",
      rc is None and len(ev(d, "launch_stale")) == 1 and not ev(d, "launch") and (out0 / "rows.jsonl").read_text() == '{"cut'
      and main(d)["stage"]["name"] == "eval0", (rc, ev(d, "launch_stale"), main(d)["stage"]))

# ---------------- overseer outage at the default waits (in-process, fake clock): every 30 min for 6 h, then exit 4;
# a PAUSE during a wait exits 3 with the decision pending
for name, pause_at in [("waits_default", None), ("waits_pause", 1)]:
    d = new(name)
    r = orch(d, "run", *GO, "--halt-before", "decide")
    run, real_t, real_once = O.Run(d), O.time, O.overseer_once
    run.c.pop("overseer_retry_waits_s")
    clk, calls = Clock(), []
    if pause_at:
        clk.sleep = lambda x, c=clk: (setattr(c, "t", c.t + x), run.pause.touch())
    O.time = clk
    O.overseer_once = lambda run, base, role, prompt, schema, st: calls.append(clk.t) or {"attempt": len(calls), "timeout": True}
    try:
        rc = O.step_stage(run, True, [])
    finally:
        O.time, O.overseer_once = real_t, real_once
    re_ = ev(d, "overseer_retry")
    stage = run.t["arms"]["main"]["stage"]["name"]
    if pause_at:
        check("outage: a PAUSE during a retry wait exits 3, decision pending, no response", rc == 3 and len(calls) == 1 and stage == "decide"
              and ev(d, "overseer_unavailable")[-1]["paused"] is True and not list((d / "overseer").glob("*.response.json")), (rc, calls, re_))
    else:
        check("outage: default waits retry every 30 min for 6 h (13 attempts), then exit 4 with the decision pending, no fallback",
              rc == 4 and O.OVERSEER_RETRY_WAITS_S == [1800] * 12 and len(calls) == 13 and calls[-1] == 6 * 3600
              and [e["wait_s"] for e in re_] == [1800] * 12 and stage == "decide" and not run.t["journal"] and not ev(d, "fallback"), (rc, calls, re_))
check("overseer pin: run/overseer_call.sh calls the pinned model at the pinned effort, with no tools",
      f"--model {O.PINNED_MODEL} --effort {O.EFFORT}" in O.OVERSEER_SH.read_text() and O.PINNED_MODEL == "claude-fable-5-2" and O.EFFORT == "xhigh"
      and '--tools ""' in O.OVERSEER_SH.read_text() and O.MAX_INPUT_TOKENS == 60000)

print("TEST_ORCH", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
