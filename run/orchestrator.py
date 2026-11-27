"""Run orchestrator, chain mode: a state machine over tree.json.

Runs in WSL with the system python3 (stdlib only). Every GPU step is a subprocess: run/trainer.py, which trains one
segment and then runs the greedy val eval in-process (eval/inproc_eval.py), under scripts/watch.sh and the eviction
sidecar, after the idle, host-RAM and gpu_procs gates.

  python3 orchestrator.py init   <run_dir> --config <run config json>
  python3 orchestrator.py run    <run_dir> [--go] [--halt-before decide] [--max-launches N] [--fake]
  python3 orchestrator.py a2     <run_dir> --seed 1|2 [--go] [--max-launches N] [--fake]
  python3 orchestrator.py status <run_dir>
  python3 orchestrator.py supervise <run_dir> [--seed 1|2] [--halt-before ...] [--max-launches N] [--fake]   (implies --go; --seed supervises A2)

Without --go nothing launches and no overseer call is made: it prints the pending stage, its command and the gates.
Exit codes: 0 finished or halted as asked, 3 paused (run_dir/PAUSE), 1 stopped on a failure (see events.log),
4 overseer unavailable (no answer after the retries, an answer that failed a model, isolation or token check twice, or
a second CONTINUE fallback in a row; the stage stays pending), 5 watch.sh's host-RAM guard killed the
launch, 6 the gate refused on RAM alone for its whole wait (ram_wait_s), 7 `a2` refused by the A2 gate (best val - base
val < 0.05), 9 `a2` seed held (listed in run_dir/A2_HOLD). It never retries a failed launch; rerunning the same command
resumes the pending stage. A nonzero trainer exit after a complete segment (its eval's done.json is written) does not
stop the run: it is logged as exit_after_segment and the segment is recorded. `supervise` reruns `run --go` after exit 5 or 6 only, at most AUTO_RESUME_MAX times per
rolling 24 h; any other exit or a PAUSE file stops it.

tree.json: "arms" holds the overseer arm ("main") and the A2 seeds ("a2s1", "a2s2"). An arm has nodes (phases), eval
points and a stage:  eval0 (the base eval) -> [ train one segment, which ends with its val eval -> decide ] ... -> done.
A train launch ends with its eval; if a PAUSE cut the eval, the stage becomes eval (an --eval-only launch). There is no
decision after the base eval (the root config trains the first segment) and none after the last eval. A2 never decides.
Units: a generation is knobs.GEN_ROLLOUTS rollouts; a node counts generations from 0; its checkpoint after generation g
is checkpoint-<g * updates per generation>. The budget is in rollouts and counts every arm node, abandoned ones too.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parent
sys.path.insert(0, str(HERE))
import knobs  # noqa: E402
import prompts  # noqa: E402
import sampler  # noqa: E402

TRAINER = HERE / "trainer.py"
OVERSEER_SH = HERE / "overseer_call.sh"
WATCH = ROOT_DIR / "scripts" / "watch.sh"
SIDECAR = ROOT_DIR / "scripts" / "evict_sidecar.sh"
GPU_PROCS = ROOT_DIR / "scripts" / "gpu_procs.ps1"
GRPO_TRAINER = ROOT_DIR / "unsloth_compiled_cache" / "UnslothGRPOTrainer.py"  # the loss code of every arm; read for its sha256 only
TEST_TASKS = ROOT_DIR / "data" / "test_tasks.txt"  # ids only, for init's disjointness assert; nothing else reads it
CLAUDE_EXE = "/mnt/c/Users/user/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe"
OVERSEER_TMP = "/mnt/c/Users/user/AppData/Local/Temp/llm-overseer-rl"
PINNED_MODEL = "claude-fable-5-2"
EFFORT = "xhigh"  # set in overseer_call.sh; init asserts the two agree
MAX_INPUT_TOKENS = 60000  # [default] fits the journal; a call above it fails the isolation check (decide: exit 4)
OVERSEER_RETRY_WAITS_S = [1800] * 12  # retry every 30 min for up to 6 h, then halt (exit 4)
EXIT_OVERSEER_UNAVAILABLE = 4
EXIT_A2_HELD = 9  # a seed listed in run_dir/A2_HOLD does not start
EXIT_A2_GATE, A2_GATE = 7, 0.05  # A2 runs only if the arm's best val beats the base eval by >= 0.05 absolute
EXIT_RAM_KILL, EXIT_RAM_GATE = 5, 6  # launch() returns their negatives (a trainer exit code can't be negative)
RAM_REASONS = {EXIT_RAM_KILL: "ram_kill", EXIT_RAM_GATE: "ram_gate"}
# 6 per 24 h (was 2). An earlier run saw 6 RAM kills in 62 launches, so a 38-hour arm
# expects about 2 per 24 h and a limit of 2 would stop it. The 12 GB launch gate still blocks a relaunch on a degraded
# host, so this limit is not the only loop guard. The 7th in 24 h stops and waits for the operator.
AUTO_RESUME_MAX, AUTO_RESUME_WINDOW_S = 6, 86400
GEN = knobs.GEN_ROLLOUTS
FAKE = {"on": False}


# ---------------------------------------------------------------- state, files, events
def atomic_write(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, obj):
    atomic_write(path, json.dumps(obj, indent=1, default=str))


class Run:
    def __init__(self, d):
        self.d = Path(os.path.expanduser(str(d))).resolve()
        self.t = json.loads((self.d / "tree.json").read_text())
        self.c = self.t["config"]

    def save(self):
        write_json(self.d / "tree.json", self.t)

    def event(self, kind, **kw):
        rec = {"time": time.strftime("%F %T"), "event": kind, **kw}
        with open(self.d / "events.log", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print("EVENT", json.dumps(rec, default=str)[:600], flush=True)

    def events(self):
        p = self.d / "events.log"
        return records(p) if p.exists() else []

    @property
    def pause(self):
        return self.d / "PAUSE"

    def node_dir(self, nid):
        return self.d / "nodes" / nid if not nid.startswith("a2s") else self.d / "baseline" / f"seed{nid[3:]}"


def complete(p):
    """A checkpoint dir whose save finished. trainer_state.json is written last (transformers 5.17.0), and a kill inside
    that write leaves it empty or cut, so it must parse too."""
    try:
        json.loads((p / "trainer_state.json").read_text())
        return True
    except (OSError, ValueError):
        return False


def ckpts(d):
    """Checkpoint steps present in a TRL output dir, complete ones only."""
    d = Path(d)
    return sorted(int(p.name.split("-")[1]) for p in d.glob("checkpoint-*") if complete(p)) if d.exists() else []


def prep_out(run, nid, out, mode, seen, done_p):
    """Repairs before a launch, run only after its gates passed, so no trainer of ours can still be writing here.
    Returns False and repairs nothing when the complete checkpoints or the eval's done.json
    differ from `seen`, what the command was built from: a trainer that outlived its orchestrator went on during the
    gate wait. The cleanup below removes incomplete dirs only, so it cannot change ckpts()."""
    if (ckpts(out), done_p.exists()) != seen:
        run.event("launch_stale", node=nid, seen=seen, now=(ckpts(out), done_p.exists()))
        return False
    if mode == "train":  # a hard kill mid-save leaves a dir TRL would resume from and fail on
        for p in out.glob("checkpoint-*"):
            if not complete(p):
                shutil.rmtree(p)
                run.event("partial_checkpoint_removed", node=nid, path=str(p))
    seal_tails(run, out)
    return True


def seal_tails(run, out):
    """End a last line that a hard kill cut, so the next launch's first appended record gets its own line instead of
    joining the fragment and being lost with it. The fragment alone is skipped by last_records."""
    for p in Path(out).glob("*.jsonl"):
        with open(p, "rb+") as f:
            size = f.seek(0, 2)
            if size:
                f.seek(size - 1)
                if f.read(1) != b"\n":
                    f.write(b"\n")
                    run.event("cut_line_ended", path=str(p))


# ---------------------------------------------------------------- val summary
def cluster_se(by_fn, nboot=2000):
    """SE of the mean of the values in by_fn {source function: [values]}: cluster bootstrap over source functions."""
    fns = sorted(by_fn)
    rng = random.Random(0)
    boots = []
    for _ in range(nboot):
        pick = [by_fn[fns[rng.randrange(len(fns))]] for _ in fns]
        boots.append(sum(map(sum, pick)) / sum(map(len, pick)))
    m = sum(boots) / nboot
    return (sum((b - m) ** 2 for b in boots) / (nboot - 1)) ** 0.5


def paired(traj_a, traj_b, nboot=2000):
    """b minus a on the val tasks both scored (joined by task_id), with the paired SE: a cluster bootstrap of the
    per-task differences over source functions. Rows of one task (greedy: one per task) are averaged first."""
    def load(p):
        d = {}
        for ln in open(p):
            r = json.loads(ln)
            d.setdefault(r["task_id"], [r["base"], []])[1].append(r["reward"])
        return {k: (b, sum(v) / len(v)) for k, (b, v) in d.items()}
    a, b = load(traj_a), load(traj_b)
    common = sorted(set(a) & set(b))
    assert common, (traj_a, traj_b)
    by_fn = {}
    for k in common:
        by_fn.setdefault(a[k][0], []).append(b[k][1] - a[k][1])
    return {"delta": round(sum(map(sum, by_fn.values())) / len(common), 4), "se": round(cluster_se(by_fn, nboot), 4),
            "n": len(common), "functions": len(by_fn), "a": str(traj_a), "b": str(traj_b)}


def val_summary(traj, nboot=2000):
    rows = [json.loads(ln) for ln in open(traj)]
    by_fn = {}
    for r in rows:
        by_fn.setdefault(r["base"], []).append(r["reward"])
    fns = sorted(by_fn)
    n = len(rows)
    p = sum(r["reward"] for r in rows) / n
    se = cluster_se(by_fn, nboot)
    ends = {}
    for r in rows:
        ends[r["end"]] = ends.get(r["end"], 0) + 1
    return {"n": n, "functions": len(fns), "passed": int(sum(r["reward"] for r in rows)), "pass_rate": round(p, 4),
            "se": round(se, 4), "ends": ends, "submitted": round(sum(bool(r["submitted"]) for r in rows) / n, 4),
            "mean_turns": round(sum(r["turns"] for r in rows) / n, 2),
            "rewards_sha": hashlib.sha256(json.dumps(sorted((r["task_id"], r["reward"]) for r in rows)).encode()).hexdigest()[:16]}


def records(path):
    """A jsonl file's records, skipping a line cut by a hard kill (train_summary's counts and joins catch anything missing)."""
    out = []
    for ln in open(path, errors="replace"):
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def last_records(path, key):
    """A jsonl file's records, the last one per key (a generation redone after a crash: its last records win)."""
    return {key(r): r for r in records(path)}


def train_summary(out, n, g0, g1, hard):
    """Generations [g0, g1) of node n from the trainer's rows.jsonl, trajectories.jsonl, steps.jsonl and summary.jsonl.
    Returns (the segment's training summary, the sampler's visits [(task_id, [its G training rewards])]).
    Training rollouts only: nothing here reads val. hard: the hard pool's task ids."""
    c = n["config"]
    G, P, U = c["num_generations"], knobs.prompts_per_gen(c), knobs.updates_per_gen(c)
    rows = last_records(out / "rows.jsonl", lambda r: (r["gen"], r["row"]))
    rows = [rows.get((g, j)) for g in range(g0, g1) for j in range(GEN)]
    assert all(rows), f"{out}/rows.jsonl: {sum(r is None for r in rows)} rows of generations {g0}..{g1 - 1} are missing"
    # rows are in dataset order (prompt index = row // G): the assigned rows, each G times
    assert [r["task_id"] for r in rows] == [t for t in n["rows"][g0 * P:g1 * P] for _ in range(G)], \
        f"{out}/rows.jsonl: task ids of generations {g0}..{g1 - 1} differ from the assigned rows"
    # Known limit: parses the node's whole trajectories.jsonl at every segment end (24,000 records at most); index it if slow
    traj = last_records(out / "trajectories.jsonl", lambda r: (r["step"], r["index"]))
    tr = [traj.get((r["step"], r["env_index"])) for r in rows]
    bad = [(r["gen"], r["row"]) for r, x in zip(rows, tr)
           if x is None or x["task_id"] != r["task_id"] or abs(x["reward"] - r["reward"]) > 1e-6 or x["passed"] != r["passed"]]
    assert not bad, f"{out}: rows.jsonl and trajectories.jsonl disagree (task, reward or passed) at (gen, row) {bad[:8]}"
    N = len(rows)

    def share(flags):
        return round(sum(map(bool, flags)) / N, 4)

    def mean(v):
        v = [x for x in v if isinstance(x, (int, float))]
        return round(sum(v) / len(v), 6) if v else None

    def ran(x):  # the submission loaded and the hidden tests ran to the end
        g = x.get("grade")
        return bool(x["submitted"]) and isinstance(g, dict) and not any(g.get(k) for k in ("error", "loaderr", "timeout", "oom"))

    groups = [rows[i:i + G] for i in range(0, N, G)]  # one prompt's G rollouts
    visits = [(g[0]["task_id"], [r["reward"] for r in g]) for g in groups]
    by_pool = {p: [r for r in rows if (r["task_id"] in hard) == (p == "hard")] for p in sampler.POOLS}
    # keyed on "loss" too: HF's end-of-train line has no loss and the same step as the launch's last update
    logs = last_records(out / "steps.jsonl", lambda r: (r["step"], "loss" in r)) if (out / "steps.jsonl").exists() else {}
    logs = [r for (s, has_loss), r in sorted(logs.items()) if g0 * U < s <= g1 * U and has_loss]

    def update(i):  # the i-th optimizer update of each generation (global step s is update (s - 1) % U)
        ls = [r for r in logs if (r["step"] - 1) % U == i]
        return {k: mean(r.get(src) for r in ls) for k, src in (("loss", "loss"), ("grad_norm", "grad_norm"),
                                                               ("clip", "clip_ratio/region_mean"), ("entropy", "entropy"))} if ls else None

    s = {"rollouts": N, "reward": mean(r["reward"] for r in rows), "pass": share(r["passed"] == 1 for r in rows),
         "partial": share(0 < r["reward"] < 1 for r in rows), "no_submit": share(not x["submitted"] for x in tr),
         "wrong": share(ran(x) and r["passed"] != 1 for r, x in zip(rows, tr)),
         "load_error": share(x["submitted"] and isinstance(x.get("grade"), dict) and x["grade"].get("loaderr") for x in tr),
         "trunc": share(r["masked_truncated"] for r in rows), "tool_calls": mean(len(x["calls"]) for x in tr),
         "zero_groups": round(sum(len(set(v)) == 1 for _, v in visits) / len(visits), 4), "len": mean(r["completion"] for r in rows),
         "pass_by_pool": {p: round(sum(r["passed"] == 1 for r in v) / len(v), 4) if v else None for p, v in by_pool.items()},
         "rollouts_by_pool": {p: len(v) for p, v in by_pool.items()},
         "updates": {"first": update(0), "second": update(1) if U > 1 else None},
         "lr_end": logs[-1].get("learning_rate") if logs else None, "steps_logged": len(logs),
         "rollout_logps_batches": max((r.get("rollout_logps_as_old_batches") or 0 for r in logs), default=None),
         "rollout_logps_fallbacks": max((r.get("rollout_logps_as_old_fallbacks") or 0 for r in logs), default=None)}
    # every launch that trained into this segment appends one summary line; its peak covers the whole launch
    # appended after the eval's done.json, so a kill can cut it on a segment the done.json recovery records
    sums = records(out / "summary.jsonl") if (out / "summary.jsonl").exists() else []
    peaks = [x["peak_reserved_GiB"] for x in sums if g0 * U < x.get("global_step", 0) <= g1 * U and x.get("peak_reserved_GiB")]
    s["peak_reserved_GiB"] = max(peaks) if peaks else None
    s["vram_headroom_MiB"] = round((knobs.VRAM_CAP_GIB - s["peak_reserved_GiB"]) * 1024) if peaks else None
    return s, visits


# ---------------------------------------------------------------- gates and launches
def sh(cmd, timeout=30):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.replace("\r", "").strip()
    except subprocess.TimeoutExpired:
        return ""


def gates(run):
    """The idle check and the host-RAM gate, plus gpu_procs on the 4060. Returns (ok, record)."""
    if FAKE["on"]:
        return True, {"fake": True}
    me = {os.getpid(), os.getppid()}
    win = sh("powershell.exe -NoProfile -Command \"[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)\"", 25)
    smi = sh("/mnt/c/Windows/System32/nvidia-smi.exe --query-gpu=memory.used --format=csv,noheader,nounits")
    apps = sh("nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader")
    py = our_python(sh("pgrep -af python").splitlines(), me)
    # Adapter LUIDs change across reboots, so the 4060's own process list comes from Windows nvidia-smi (graphics and
    # compute); gpu_procs is logged for the record.
    full = sh("/mnt/c/Windows/System32/nvidia-smi.exe")
    others = full.split("Processes:", 1)[-1].strip() if "No running processes found" not in full else ""
    procs = sh(f"powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"$(wslpath -w {GPU_PROCS})\"", 25)
    rec = {"windows_free_GB": win, "gpu_used_MiB": smi, "compute_apps": apps, "wsl_python": py, "gpu_procs": procs,
           "procs_on_4060": others[-1500:]}
    def num(x):
        try:
            return float(x)
        except ValueError:
            return None
    refused = [why for why, bad in (
        ("gpu_compute_apps", apps), ("wsl_python", py), ("gpu_used", num(smi) is None or num(smi) > 100),
        ("nvidia_smi", "Processes:" not in full), ("procs_on_4060", others),
        ("ram", num(win) is None or num(win) < run.c.get("min_free_gb", 12.0))) if bad]
    rec["refused"] = refused
    return not refused, rec


OUR_PY = re.compile(r"grpo_dummy|run/trainer\.py|evalharness|inproc_eval|vllm|launcher\.py|server\.py|grade\.py")


def our_python(lines, me):
    """WSL python processes that would share the GPU or host RAM with a launch: the trainer, the eval, vLLM and the
    sandbox (launcher, server, grader). Other python (an editor's server, a shell running find) does not block."""
    return [ln for ln in lines if ln.split() and OUR_PY.search(ln) and "pgrep" not in ln and int(ln.split()[0]) not in me]


def wait_idle(run):
    """Wait for the gates; the GPU may still be releasing the last run. Up to idle_wait_s (default 15 min), polling every
    30 s; when host RAM is the only refusal, up to ram_wait_s (default 2 h), every 60 s (idle free RAM recovers slowly
    after a vLLM run). Every refusal is logged with its reasons."""
    t0 = time.time()
    while True:
        ok, rec = gates(run)
        waited = round(time.time() - t0)
        if ok:
            if waited:
                run.event("gates_passed", waited_s=waited, windows_free_GB=rec.get("windows_free_GB"))
            return rec
        ram_only = rec["refused"] == ["ram"]
        run.event("gates_wait", refused=rec["refused"], waited_s=waited, windows_free_GB=rec.get("windows_free_GB"),
                  wsl_python=rec.get("wsl_python"), gpu_used_MiB=rec.get("gpu_used_MiB"))
        if waited > (run.c.get("ram_wait_s", 7200) if ram_only else run.c.get("idle_wait_s", 900)):
            run.event("gates_refused", **rec)
            run.refused = rec["refused"]
            return None
        time.sleep(60 if ram_only else 30)


def trainer_sha():
    """sha256 of the compiled GRPO trainer. It is outside the manifest (Unsloth may regenerate it), so every launch
    records it, before and after: status warns when a run saw more than one value."""
    return hashlib.sha256(GRPO_TRAINER.read_bytes()).hexdigest() if GRPO_TRAINER.exists() else None


def launch(run, name, argv, stage, prep):
    """One GPU subprocess under watch.sh and the sidecar. Returns its exit code (97 watchdog kill, 98 no baseline).
    prep() runs once the gates have passed, just before the trainer starts; when it returns False the command is stale,
    nothing is launched, and this returns None."""
    logdir = run.d / "logs"
    logdir.mkdir(exist_ok=True)
    if FAKE["on"]:
        if not prep():
            return None
        rc = subprocess.run([sys.executable, str(ROOT_DIR / "tests" / "fake_train.py"), *argv[2:]],
                            stdout=open(logdir / f"{name}.log", "w"), stderr=subprocess.STDOUT).returncode
        run.event("launch_end", name=name, stage=stage, exit=rc, trainer_sha256=trainer_sha())
        return -EXIT_RAM_KILL if ram_killed(rc, logdir / f"{name}.log") else rc
    idle = wait_idle(run)
    if idle is None:
        return -EXIT_RAM_GATE if run.refused == ["ram"] else -1
    if not prep():
        return None
    run.event("launch", name=name, stage=stage, gates=idle, argv=argv, trainer_sha256=trainer_sha())
    rundir = os.path.relpath(logdir, os.path.expanduser("~/llm-overseer-rl/runs"))
    log = logdir / f"{name}.log"
    # whole_run: a launch trains ~65 min, so keep evicting page cache and tracing host RAM
    # past the first step, for the RAM kills we expect
    side = subprocess.Popen(["bash", str(SIDECAR), str(log), "whole_run"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    env = {**os.environ, "RUNDIR": rundir, "SPILL_GB": "0.3", "POLL": "10", "PAUSE_FILE": str(run.pause)}
    rc = subprocess.run(["bash", str(WATCH), name, " ".join(argv)], env=env, cwd=str(WATCH.parent)).returncode
    try:
        side.wait(timeout=120)
    except subprocess.TimeoutExpired:
        side.kill()
    time.sleep(5)
    ok, after = gates(run)
    run.event("launch_end", name=name, stage=stage, exit=rc, gpu_idle_after=ok, after=after, trainer_sha256=trainer_sha())
    return -EXIT_RAM_KILL if ram_killed(rc, log) else rc


def ram_killed(rc, log):
    """watch.sh exits 97 for every kill (RAM, silent log, no exit after PAUSE); only the host-RAM guard's may auto-resume."""
    return rc == 97 and log.exists() and "WATCHDOG_KILL host free RAM" in log.read_text(errors="replace")


def trainer_argv(run, arm, nid, seg_end, mode):
    """mode: train (fresh, --init-adapter [--init-optimizer] or --resume, decided from the node's dir) or eval (--eval-only)."""
    A = run.t["arms"][arm]
    n = A["nodes"][nid]
    nd = run.node_dir(nid)
    nd.mkdir(parents=True, exist_ok=True)
    out = nd / "checkpoints"
    write_json(nd / "config.json", n["config"])
    write_json(nd / "stream.json", {"run": run.t["run_id"], "name": n["stream"], "P": knobs.prompts_per_gen(n["config"]), "rows": n["rows"]})
    c = run.c
    # The budget can be extended mid-run; trainer_budget pins --max-gens (the dataset length) so resumes stay consistent
    argv = ["python", str(TRAINER), "--model", c["model"], "--unsloth", "--vllm", "--vllm-util", str(c.get("vllm_util", 0.85)),
            "--mem-frac", str(c.get("mem_frac", 0.99)), "--env", "code", "--tasks", c["tasks_dir"],
            "--node-config", str(nd / "config.json"), "--stream", str(nd / "stream.json"),
            "--max-gens", str(A.get("trainer_budget", A["budget"]) // GEN), "--save-gens", str(c["save_gens"]), "--out", str(out),
            "--val-tasks", c["val_tasks"], "--val-out", str(nd / "val" / str(seg_end)), "--pause-file", str(run.pause),
            "--seed", str(n["seed"])]
    if mode == "eval":
        ck = out / f"checkpoint-{seg_end * knobs.updates_per_gen(n['config'])}"
        argv += ["--eval-only"] + (["--init-adapter", str(ck)] if seg_end else [])
    else:
        argv += ["--stop-at-gen", str(seg_end)]
        start = A["evals"][n["from"]]["ckpt"] if n["from"] else None  # None: the base model
        if ckpts(out):
            argv += ["--resume"]
        elif start:  # a phase's first launch: the start point's LoRA, and its optimizer state unless the phase resets it
            argv += ["--init-adapter", start] + ([] if n["config"]["reset_optimizer"] else ["--init-optimizer", start])
    return argv


# ---------------------------------------------------------------- overseer
def isolation(raw, j):
    """What the call could see and do, from overseer_call.sh's stream log. A clean call is one API request (num_turns 2:
    the answer goes through the StructuredOutput tool), no other tool, no MCP server."""
    stream = raw.with_name(raw.name[:-len(".json")] + ".stream.jsonl")
    recs = [json.loads(ln) for ln in open(stream, encoding="utf-8") if ln.strip().startswith("{")] if stream.exists() else []
    init = next((x for x in recs if x.get("type") == "system" and x.get("subtype") == "init"), None)
    u = j.get("usage") or {}
    iters = u.get("iterations")
    used = sorted({c.get("name") for x in recs if x.get("type") == "assistant"
                   for c in (x.get("message") or {}).get("content") or [] if c.get("type") == "tool_use"})
    iso = {"num_turns": j.get("num_turns"), "api_requests": len(iters) if isinstance(iters, list) else None,
           "tools": init.get("tools") if init else None, "mcp_servers": init.get("mcp_servers") if init else None,
           "tools_used": used, "input_tokens": sum(u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                                          "cache_read_input_tokens")),
           "problems": []}
    pr = iso["problems"]
    if init is None:
        pr.append("no init record in the stream log")
    else:
        if set(iso["tools"] or []) - {"StructuredOutput"}:
            pr.append(f"tools available: {iso['tools']}")
        if iso["mcp_servers"]:
            pr.append(f"MCP servers: {iso['mcp_servers']}")
    if iso["num_turns"] != 2 or iso["api_requests"] != 1:
        pr.append(f"num_turns {iso['num_turns']}, API requests {iso['api_requests']} (expected 2 and 1)")
    if set(used) - {"StructuredOutput"}:
        pr.append(f"tools used: {used}")
    if iso["input_tokens"] > MAX_INPUT_TOKENS:  # a prompt over the cap is an error, as is an unexplained large call
        pr.append(f"input tokens {iso['input_tokens']} > {MAX_INPUT_TOKENS}")
    iso["claude_code_version"] = init.get("claude_code_version") if init else None
    return iso


def unavailable(r):
    """Why no answer arrived, or None if one did. No answer: a timeout, the claude -p exit or the
    wrapper's own exit nonzero, is_error (usage limit, auth, network, a broken update), no parsable output. These are
    retried and then halt the run. An answer that arrived but fails a check is invalid: decide() retries it once, then
    halts (model, isolation, tokens) or falls back to CONTINUE (the answer's content)."""
    if r["timeout"]:
        return "timeout"
    if r.get("exit") not in (0, None):
        return f"wrapper exit {r['exit']}"
    m = re.search(r"exit=(\d+)", r.get("wrapper_out") or "")
    if m and m.group(1) != "0":
        return f"claude exit {m.group(1)}"
    if r.get("no_output"):
        return r["no_output"]
    if r.get("is_error"):
        return f"is_error: {(r.get('result_text') or '')[:200]}"
    return None


def overseer(run, st, role, prompt, schema):
    """Request on disk, call, response on disk; then the caller acts. A request without a response is re-sent.
    A call with no answer (unavailable()) is retried after each of overseer_retry_waits_s; if none answers, the response
    is not written and r["unavailable"] is set: the caller halts with the stage pending and a rerun re-sends the call.
    st: the pending stage (it keeps the call number)."""
    od = run.d / "overseer"
    od.mkdir(exist_ok=True)
    if "call" not in st:  # a new call; a resumed stage keeps its number and re-sends
        run.t["calls"] += 1
        st["call"] = run.t["calls"]
        run.save()
    base = od / f"{st['call']:03d}_{role}"
    resp = base.with_name(base.name + ".response.json")
    if resp.exists():
        return json.loads(resp.read_text())
    waits = list(run.c.get("overseer_retry_waits_s", OVERSEER_RETRY_WAITS_S))
    while True:
        r = overseer_once(run, base, role, prompt, schema, st)
        why = unavailable(r)
        if why is None:
            break
        r["unavailable"] = why
        write_json(base.with_name(f"{base.name}.attempt{r['attempt']}.unavailable.json"), r)  # kept; resp is not written
        w = waits.pop(0) if waits and not run.pause.exists() else None
        if w is not None:
            run.event("overseer_retry", call=st["call"], role=role, attempt=r["attempt"], why=why, wait_s=w)
            t0 = time.time()
            while time.time() - t0 < w and not run.pause.exists():
                time.sleep(min(5, w))
        if w is None or run.pause.exists():  # out of retries, or a PAUSE (before or during the wait): halt, no further call
            run.event("overseer_unavailable", call=st["call"], role=role, attempt=r["attempt"], why=why,
                      paused=run.pause.exists())
            return r
    write_json(resp, r)
    return r


def overseer_once(run, base, role, prompt, schema, st):
    req = base.with_name(base.name + ".request.json")
    attempt = json.loads(req.read_text())["attempt"] + 1 if req.exists() else 1
    if attempt > 1:
        run.event("overseer_resend", call=st["call"], role=role, attempt=attempt)
    write_json(req, {"role": role, "attempt": attempt, "time": time.strftime("%F %T"), "model": PINNED_MODEL, "effort": EFFORT,
                     "prompt_version": prompts.PROMPT_VERSION, "prompt": prompt, "schema": schema})
    atomic_write(base.with_name(base.name + ".prompt.txt"), prompt)
    write_json(base.with_name(base.name + ".schema.json"), schema)
    raw = base.with_name(base.name + ".raw.json")
    env = {**os.environ}
    env.setdefault("CLAUDE_BIN", CLAUDE_EXE)
    env.setdefault("OVERSEER_TMP", OVERSEER_TMP)
    env.setdefault("OVERSEER_STDIN", "1")  # prompt on stdin, clear of the 32K Windows command line
    Path(env["OVERSEER_TMP"]).mkdir(parents=True, exist_ok=True)
    cmd = (os.environ.get("OVERSEER_CMD", f"bash {OVERSEER_SH}").split()
           + [str(base.with_name(base.name + ".prompt.txt")), str(base.with_name(base.name + ".schema.json")), str(raw)])
    t0 = time.time()
    r = {"role": role, "call": st["call"], "attempt": attempt, "timeout": False}
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        r["wrapper_out"] = p.communicate(timeout=run.c.get("overseer_timeout_s", 1800))[0][-2000:]
        r["exit"] = p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        r["timeout"], r["exit"] = True, None
        for w in Path(env["OVERSEER_TMP"]).iterdir():  # the killed wrapper never reached its rmdir; calls are sequential
            if w.name.startswith("tmp."):
                shutil.rmtree(w, ignore_errors=True)
    r["secs"] = round(time.time() - t0, 1)
    j = None
    try:
        j = json.loads(raw.read_text())
        r.update(structured_output=j.get("structured_output"), is_error=j.get("is_error"), cost_usd=j.get("total_cost_usd"),
                 usage=j.get("usage"), models=sorted((j.get("modelUsage") or {}).keys()), result_text=(j.get("result") or "")[:2000])
    except (OSError, ValueError) as e:
        r["no_output"] = f"no parsable output: {e!r}"[:300]
    if unavailable(r) is None:  # an answer arrived: check it; a failed check is an invalid answer (fallbacks)
        if r["models"] != [PINNED_MODEL]:
            r["error"] = f"model mismatch: {r['models']}"  # never act on another model's answer, or on an unnamed one
        else:
            r["isolation"] = iso = isolation(raw, j)
            if iso["problems"]:
                r["error"] = "isolation: " + "; ".join(iso["problems"])
    run.event("overseer", call=st["call"], role=role, attempt=attempt, secs=r["secs"], timeout=r["timeout"],
              cost_usd=r.get("cost_usd"), input_tokens=(r.get("isolation") or {}).get("input_tokens"),
              unavailable=unavailable(r), error=r.get("error"))
    return r


# ---------------------------------------------------------------- chain logic
def new_node(nid, frm, config, stream, seed, smp):
    """A phase: trains from eval point `frm` (None: the base model) with one config. gens counts its finished
    generations; rows are its assigned task ids; sampler is its live sampler state (snapshotted at every eval point)."""
    return {"id": nid, "from": frm, "config": config, "stream": stream, "seed": seed, "gens": 0, "rows": [], "sampler": smp,
            "status": "running", "created": time.strftime("%F %T")}


def pools_of(run):
    if not hasattr(run, "pools"):
        run.pools = {p: [ln.strip() for ln in open(run.t["pools"][p]["path"]) if ln.strip()] for p in sampler.POOLS}
    return run.pools


def seg_gens(run):
    return run.c["eval_every_rollouts"] // GEN


def ensure_rows(run, n, seg_end):
    """Assign node n's rows up to generation seg_end (before each segment). Moves the sampler's cursors;
    its clock and the zero-signal rule move in finish_segment."""
    P = knobs.prompts_per_gen(n["config"])
    k = seg_end - len(n["rows"]) // P
    if k > 0:
        n["rows"] += sampler.assign(n["sampler"], pools_of(run), P, n["config"]["hard_share"], k)
        run.save()


def path(A, eid):
    """Eval ids from the base eval to eid."""
    out = []
    while eid:
        out.append(eid)
        eid = A["evals"][eid]["prev"]
    return out[::-1]


def trail3(A, eid):
    """Mean val pass rate of eid and the two evals before it on its path; None if the path is shorter than 3."""
    p = path(A, eid)[-3:]
    v = [A["evals"][e]["val"] for e in p]
    # from pass counts, not the rounded pass rates, so equal counts tie exactly and the earliest wins
    return round(sum(x["passed"] for x in v) / sum(x["n"] for x in v), 6) if len(p) == 3 else None


def final_pick(A):
    """[default] The same function for every arm: the eval point with the highest trail-3 score among
    those with at least 3 evals on their path from the base (the base eval counts); ties go to the earliest."""
    best = None
    for eid in A["eval_order"]:
        e = A["evals"][eid]
        if e["trail3"] is not None and (best is None or e["trail3"] > best["trail3"]):
            best = e
    return best and {**{k: best[k] for k in ("id", "node", "gen", "trail3", "ckpt")}, "pass_rate": best["val"]["pass_rate"]}


def finish_segment(run, arm, nid, seg_end, kind):
    """After a launch exit 0: record the eval point (val summary, checkpoint, sampler snapshot, training summary), apply
    the zero-signal rule, count the rollouts, prune checkpoints. Everything lands in tree.json with the caller's one
    save, so a kill before it leaves the segment unrecorded and a rerun records it once (done.json recovery)."""
    A = run.t["arms"][arm]
    n = A["nodes"][nid]
    nd = run.node_dir(nid)
    out = nd / "checkpoints"
    vdir = nd / "val" / str(seg_end)
    assert (vdir / "done.json").exists(), f"{vdir} has no done.json"
    assert not any(e["node"] == nid and e["gen"] == seg_end for e in A["evals"].values()), f"{nid} generation {seg_end} is already an eval point"
    traj = vdir / "traj.jsonl"
    v = val_summary(traj)
    write_json(nd / "val" / f"{seg_end}.json", v)
    U, g0, train = knobs.updates_per_gen(n["config"]), n["gens"], None
    if seg_end > g0:
        assert seg_end * U in ckpts(out), f"{out} has no checkpoint-{seg_end * U}"
        train, visits = train_summary(out, n, g0, seg_end, set(pools_of(run)["hard"]))
        # The trainer asserts this itself (SEGMENT_CHECK), but after its final save: a rerun then finds the end
        # checkpoint, runs the eval and would record the segment. Asserted here so no rerun gets past it.
        assert train["rollout_logps_fallbacks"] == 0 and train["rollout_logps_batches"], \
            (f"{nid} generations {g0}..{seg_end - 1}: rollout logprobs as the old policy: fallbacks "
             f"{train['rollout_logps_fallbacks']}, batches {train['rollout_logps_batches']} (expected 0 and above 0)")
        sampler.update(n["sampler"], visits, seg_end - g0)
        A["rollouts_used"] += (seg_end - g0) * GEN  # per finished segment: a generation redone after a crash counts once
        n["gens"] = seg_end
    # the base eval's snapshot is the untouched state (the root's first rows are assigned before it runs)
    snap = copy.deepcopy(n["sampler"]) if seg_end else sampler.new(run.t["run_id"], n["stream"])
    mine = [e for e in A["eval_order"] if A["evals"][e]["node"] == nid]
    prev = mine[-1] if mine else n["from"]
    eid = f"e{len(A['eval_order'])}"
    e = {"id": eid, "node": nid, "gen": seg_end, "prev": prev, "lineage_rollouts": snap["rollouts"], "val": v, "traj": str(traj),
         "ckpt": str(out / f"checkpoint-{seg_end * U}") if seg_end else None, "sampler": snap, "train": train,
         "pools": sampler.counts(snap, pools_of(run)),
         "vs_base": paired(A["evals"]["e0"]["traj"], traj) if A["eval_order"] else None,
         "vs_prev": paired(A["evals"][prev]["traj"], traj) if prev else None,
         "after_decision": len(run.t["journal"]) or None if arm == "main" else None, "time": time.strftime("%F %T")}
    A["evals"][eid] = e
    A["eval_order"].append(eid)
    e["trail3"] = trail3(A, eid)
    keep = seg_gens(run) * U
    for s in ckpts(out):  # eval points are never pruned; the save_gens ones between them are
        if s < seg_end * U and s % keep:
            shutil.rmtree(out / f"checkpoint-{s}")
    hr = (train or {}).get("vram_headroom_MiB")
    if hr is not None and hr < knobs.VRAM_BAR_MIB:  # logged only, no action: the next launch could OOM
        run.event("vram_margin_low", node=nid, gen=seg_end, headroom_MiB=hr, bar_MiB=knobs.VRAM_BAR_MIB)
    run.event("segment_done", arm=arm, node=nid, gen=seg_end, eval=eid, via=kind, pass_rate=v["pass_rate"], se=v["se"],
              delta_vs_base=(e["vs_base"] or {}).get("delta"), delta_vs_prev=(e["vs_prev"] or {}).get("delta"),
              paired_se=(e["vs_prev"] or {}).get("se"), trail3=e["trail3"], reward=(train or {}).get("reward"),
              rollouts_used=A["rollouts_used"])


def next_stage(run, arm, nid):
    A = run.t["arms"][arm]
    n = A["nodes"][nid]
    if A["rollouts_used"] >= A["budget"]:  # the budget is always spent in full; no decision follows the last eval
        n["status"] = "done"
        run.event("arm_done", arm=arm, rollouts_used=A["rollouts_used"], evals=len(A["eval_order"]), final_pick=final_pick(A))
        return {"name": "done"}
    if arm != "main" or n["gens"] == 0:  # A2 never decides; the root's first segment follows the base eval directly
        return {"name": "train", "node": nid, "seg_end": n["gens"] + seg_gens(run)}
    return {"name": "decide", "eval": A["eval_order"][-1]}


def sync_journal(run):
    """journal.jsonl (append-only) mirrors tree.json's journal, which is saved together with the stage it leads to."""
    p = run.d / "journal.jsonl"
    have = sum(1 for _ in open(p)) if p.exists() else 0
    with open(p, "a") as f:
        for j in run.t["journal"][have:]:
            f.write(json.dumps(j) + "\n")


def decide(run, go):
    """One decision at the overseer arm's tip. An unavailable overseer halts with the stage pending,
    never a fallback. An invalid answer is retried once (with the validation error, if the answer itself was wrong).
    Two content errors (wrong type, wrong enum, bad target) fall back to CONTINUE, journaled with fallback true, unless
    the previous decision was a fallback too. That case, and a call that failed a model, isolation or token check on
    either try, halt with exit 4 and the stage pending: silent fallbacks would turn this arm into A2."""
    t = run.t
    A = t["arms"]["main"]
    st = A["stage"]
    tip = A["evals"][A["eval_order"][-1]]
    assert st["eval"] == tip["id"], (st, tip["id"])
    if not go:
        print("DRY next: decision call at", tip["id"], "rollouts used", A["rollouts_used"], "of", A["budget"])
        return 0
    targets = A["eval_order"][:-1]  # REVERT names an earlier eval point; the tip would be a NEW_PHASE
    d = None
    while d is None:
        r = overseer(run, st, "decide", prompts.decision(t, st.get("error"), MAX_INPUT_TOKENS), knobs.decision_schema(targets))
        if r.get("unavailable"):  # no answer: halt with the stage (its tries and error) pending
            run.save()
            return 3 if run.pause.exists() else EXIT_OVERSEER_UNAVAILABLE
        so, clamps = r.get("structured_output"), []
        if r.get("error") or so is None:  # the call failed a check (model, isolation, tokens): nothing to tell the overseer
            err, feedback = r.get("error") or "no structured output", None
        else:
            d, clamps, err = knobs.check_decision(so, targets, "e0")
            feedback = err
        if d is None:
            st["tries"] = st.get("tries", 0) + 1
            run.event("decision_invalid", call=r["call"], error=err, tries=st["tries"])
            if st["tries"] >= 2:
                check_failed = feedback is None or st.get("error") is None  # this try or the first (its feedback is st["error"])
                if check_failed or (t["journal"] and t["journal"][-1]["fallback"]):
                    run.event("decision_halt", call=r["call"], error=err,
                              why="a call failed a model, isolation or token check" if check_failed else "second fallback in a row")
                    for k in ("call", "tries", "error"):
                        st.pop(k, None)  # a rerun takes this decision again with a new call
                    run.save()
                    return EXIT_OVERSEER_UNAVAILABLE
                break
            st.pop("call")  # a new call
            st["error"] = feedback
            run.save()
    fallback = d is None
    if fallback:
        d, clamps = {"decision": "CONTINUE", **{k: "" for k in knobs.TEXT_FIELDS}}, []
        run.event("fallback", call=r["call"], why=err, action="CONTINUE")
    for cl in clamps:
        run.event("clamp", call=r["call"], **cl)
    cur = A["nodes"][tip["node"]]
    node = cur
    if d["decision"] != "CONTINUE":  # NEW_PHASE: from the tip; REVERT: from the target. Sampler state: the start point's snapshot
        start = A["evals"][d.get("target", tip["id"])]
        node = new_node(f"n{len(A['order'])}", start["id"], d["config"], cur["stream"], cur["seed"], copy.deepcopy(start["sampler"]))
        A["nodes"][node["id"]] = node
        A["order"].append(node["id"])
        cur["status"] = "left"
    j = {"index": len(t["journal"]) + 1, "eval": tip["id"], "rollouts_used": A["rollouts_used"], "decision": d["decision"],
         "target": d.get("target"), "node": node["id"], "config": node["config"],
         "changed": {k: [cur["config"][k], node["config"][k]] for k in knobs.knob_names()
                     if k != "reset_optimizer" and node["config"][k] != cur["config"][k]},
         **{k: d[k] for k in knobs.TEXT_FIELDS}, "fallback": fallback, "clamps": clamps,
         "overseer": {"call": r["call"], "attempt": r["attempt"], "models": r.get("models"), "effort": EFFORT,
                      "input_tokens": (r.get("isolation") or {}).get("input_tokens"),
                      "output_tokens": (r.get("usage") or {}).get("output_tokens"), "secs": r["secs"],
                      "cost_usd": r.get("cost_usd"), "prompt_version": prompts.PROMPT_VERSION},
         "time": time.strftime("%F %T")}
    t["journal"].append(j)
    A["stage"] = {"name": "train", "node": node["id"], "seg_end": node["gens"] + seg_gens(run)}
    run.save()  # the journal entry, the new node and the stage in one write: a decision is never taken twice
    sync_journal(run)
    run.event("decision", index=j["index"], eval=tip["id"], decision=j["decision"], target=j["target"], node=node["id"],
              changed=j["changed"], fallback=fallback, call=r["call"])
    return None


def step_stage(run, go, halt_before, arm="main"):
    """Do the arm's pending stage once. Returns None to go on, or an exit code."""
    A = run.t["arms"][arm]
    st = A["stage"]
    name = st["name"]
    if name == "done":
        print("ARM DONE", arm, json.dumps({"rollouts_used": A["rollouts_used"], "budget": A["budget"], "nodes": len(A["order"]),
                                           "evals": len(A["eval_order"]), "final_pick": final_pick(A)}))
        return 0
    if halt_before and name in halt_before:
        run.event("halt", before=name)
        return 0
    if run.pause.exists():
        run.event("paused", stage=st)
        return 3
    if name in ("eval0", "train", "eval"):
        nid, seg_end = st["node"], st["seg_end"]
        n = A["nodes"][nid]
        mode = "train" if name == "train" else "eval"
        out = run.node_dir(nid) / "checkpoints"
        end_step = seg_end * knobs.updates_per_gen(n["config"])
        done_p = run.node_dir(nid) / "val" / str(seg_end) / "done.json"
        if done_p.exists():
            # the eval finished but its stage update was lost (orchestrator killed after the trainer wrote done.json, or
            # the trainer failed in teardown): record it now; a relaunch would refuse to overwrite a finished eval
            run.event("segment_recovered", node=nid, gen=seg_end, stage=dict(st))
            finish_segment(run, arm, nid, seg_end, "recovered")
            A["stage"] = next_stage(run, arm, nid)
            run.save()
            return None
        if mode == "train" and end_step in ckpts(out):  # trained earlier, its eval was cut by a pause or a failure
            st.update(name="eval")
            run.save()
            return None
        if go:
            ensure_rows(run, n, seg_end or seg_gens(run))  # the trainer needs rows even for the base eval
        seen = (ckpts(out), done_p.exists())  # what argv is built from
        argv = trainer_argv(run, arm, nid, seg_end, mode)
        if not go:
            print("DRY next:", name, nid, seg_end, "\n  " + " ".join(argv))
            ok, rec = gates(run)
            print("  gates", "ok" if ok else "would refuse", json.dumps(rec)[:800])
            return 0
        st["launches"] = st.get("launches", 0) + 1
        run.save()
        rc = launch(run, f"{nid}_g{seg_end}_{mode}_{st['launches']}", argv, dict(st), lambda: prep_out(run, nid, out, mode, seen, done_p))
        if rc is None:
            return None  # stale: start the stage over from what is on disk now
        done = done_p.exists()
        if rc == 3:
            if not done and end_step in ckpts(out):
                st.update(name="eval")  # the segment trained; its eval was cut
            run.save()
            run.event("paused", stage=st, launch_exit=rc)
            return 3
        if rc != 0 and not done:
            why = RAM_REASONS.get(-rc)
            run.event("launch_failed", stage=st, exit=rc, **({"reason": why} if why else {}))
            return -rc if why else 1
        if rc != 0:
            # The segment is complete: the trainer's SEGMENT_CHECK ran before the eval and done.json is the eval's last
            # write, so this is a failure in teardown. Go on (the default; to make
            # it halt again, return 1 here). A rerun would record this segment through the done.json recovery anyway.
            run.event("exit_after_segment", stage=st, exit=rc)
        finish_segment(run, arm, nid, seg_end, mode)
        A["stage"] = next_stage(run, arm, nid)
        run.save()
        return None
    if name == "decide":
        assert arm == "main", "A2 never calls the overseer"
        return decide(run, go)
    raise ValueError(f"unknown stage {st}")


# ---------------------------------------------------------------- A2 baseline
def a2_start(run, seed):
    """A2 seed k: the frozen root config for the overseer arm's rollouts_used, with the same eval
    cadence, reward, sampler rule, checkpoint rule and final pick. Seed 1 continues the overseer arm's root node from
    its last eval point; seed 2 starts from the base with its own trainer seed and stream name and reuses the base
    eval. Returns the arm's name, or None if the A2 gate refuses."""
    t = run.t
    M = t["arms"]["main"]
    assert M["stage"]["name"] == "done", "A2 runs after the overseer arm finishes (its budget is that arm's rollouts_used)"
    arm = f"a2s{seed}"
    base = M["evals"]["e0"]["val"]["pass_rate"]
    best = max(M["evals"][e]["val"]["pass_rate"] for e in M["eval_order"][1:])
    gap = round(best - base, 4)
    # a2_min_gap: configs/smoke.json only (a tiny smoke budget cannot move val by 0.05, and S5 must reach A2); root.json has none
    min_gap = run.c.get("a2_min_gap", A2_GATE)
    run.event("a2_gate", seed=seed, best=best, base=base, gap=gap, min_gap=min_gap, passed=gap >= min_gap)
    if gap < min_gap:
        print(f"A2 GATE REFUSED: best {best} - base {base} = {gap} < {min_gap}")
        return None
    if arm in t["arms"]:
        return arm
    if M["order"] == ["n0"]:  # every decision was CONTINUE or a fallback: the overseer arm is the root config throughout
        fb = sum(j["fallback"] for j in t["journal"])
        run.event("a2_warning", seed=seed, why="the overseer arm never left n0", decisions=len(t["journal"]), fallbacks=fb)
        print(f"A2 WARNING: the overseer arm never left n0 ({len(t['journal'])} decisions, {fb} fallbacks). It ran the root "
              "config throughout, so A2 seed 1 is a copy of it and the comparison compares a run with itself.", flush=True)
    raw = (run.d / "root_config.json").read_bytes()
    frozen, n0 = json.loads(raw), M["nodes"]["n0"]
    assert hashlib.sha256(raw).hexdigest() == t["root_config_sha256"], "root_config.json changed since init"
    assert frozen == t["root_config"] == n0["config"], "the root node did not run the frozen root config"
    nd = run.node_dir(arm)
    (nd / "checkpoints").mkdir(parents=True, exist_ok=True)
    if seed == 1:  # same config, stream and seed as the root node: its eval points are seed 1's first ones
        ids = [e for e in M["eval_order"] if M["evals"][e]["node"] == "n0"]
        last = M["evals"][ids[-1]]
        assert last["gen"] > 0 and ids == [f"e{i}" for i in range(len(ids))], ids
        node = new_node(arm, None, frozen, n0["stream"], n0["seed"], copy.deepcopy(last["sampler"]))
        node["gens"], node["rows"] = last["gen"], n0["rows"][:last["gen"] * knobs.prompts_per_gen(frozen)]
        src = run.node_dir("n0") / "checkpoints"
        shutil.copytree(last["ckpt"], nd / "checkpoints" / Path(last["ckpt"]).name, dirs_exist_ok=True)
        for f in ("steps.jsonl", "rows.jsonl", "trajectories.jsonl"):
            shutil.copy(src / f, nd / "checkpoints" / f)
        evals = {e: dict(M["evals"][e], node=arm, from_root=True, after_decision=None) for e in ids}
        run.event("a2_seed1_from_root", gen=last["gen"], eval=last["id"])
    else:  # generation 0 is the base model either way (LoRA B starts at zero): reuse the base eval
        node = new_node(arm, None, frozen, arm, seed, sampler.new(t["run_id"], arm))
        evals = {"e0": dict(M["evals"]["e0"], node=arm, from_root=True, sampler=sampler.new(t["run_id"], arm))}
    t["arms"][arm] = {"nodes": {arm: node}, "order": [arm], "evals": evals, "eval_order": list(evals),
                      "rollouts_used": node["gens"] * GEN, "budget": M["rollouts_used"]}
    t["arms"][arm]["stage"] = next_stage(run, arm, arm)
    run.save()
    return arm


# ---------------------------------------------------------------- supervisor
def supervise(run_dir, extra, seed=None):
    """Rerun `run --go` (or `a2 --seed N --go` when seed is given) after a host-RAM kill (5) or a RAM-only gate timeout (6), at most AUTO_RESUME_MAX
    times per rolling 24 h, counted from events.log. The relaunch waits in the orchestrator's own gate (min_free_gb)."""
    while True:
        cmd = ["a2", run_dir, "--seed", str(seed)] if seed else ["run", run_dir]
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), *cmd, "--go", *extra]).returncode
        run = Run(run_dir)
        why = RAM_REASONS.get(rc)
        if why is None or run.pause.exists():
            print("SUPERVISE stop: exit", rc, "(PAUSE present)" if run.pause.exists() else "", flush=True)
            return rc
        now = time.time()
        evs = run.events()
        recent = [e for e in evs if e["event"] == "auto_resume"
                  and now - time.mktime(time.strptime(e["time"], "%Y-%m-%d %H:%M:%S")) < AUTO_RESUME_WINDOW_S]
        # An `auto_resume_grant` event {count: N} allows N more resumes after it, besides the 24 h rule
        g = max((i for i, e in enumerate(evs) if e["event"] == "auto_resume_grant"), default=None)
        granted = g is not None and sum(e["event"] == "auto_resume" for e in evs[g:]) < evs[g]["count"]
        if len(recent) >= AUTO_RESUME_MAX and not granted:
            run.event("auto_resume_refused", reason=why, count=len(recent), window_s=AUTO_RESUME_WINDOW_S)
            print("SUPERVISE stop: auto-resume limit reached, waiting for the operator", flush=True)
            return rc
        run.event("auto_resume", reason=why, windows_free_GB=gates(run)[1].get("windows_free_GB"), count=len(recent) + 1)


# ---------------------------------------------------------------- CLI
def cmd_init(d, cfg_path):
    d = Path(os.path.expanduser(d))
    assert not (d / "tree.json").exists(), f"{d}/tree.json exists"
    c = json.loads(Path(cfg_path).read_text())
    c["tasks_dir"] = os.path.expanduser(c["tasks_dir"])
    for k in ("pool", "hard_pool", "val_tasks"):  # relative to the repo root; an absolute path stays as it is
        c[k] = str(ROOT_DIR / c[k])

    def ids(p):
        return [ln.strip() for ln in open(p) if ln.strip()]

    sets = {"easy": ids(c["pool"]), "hard": ids(c["hard_pool"]), "val": ids(c["val_tasks"]), "test": ids(TEST_TASKS)}
    for x, y in (("easy", "hard"), ("easy", "val"), ("hard", "val"), ("easy", "test"), ("hard", "test"), ("val", "test")):
        assert not set(sets[x]) & set(sets[y]), f"{x} and {y} overlap"
    pools = {}
    for p, key in (("easy", "pool"), ("hard", "hard_pool")):
        sha = hashlib.sha256("".join(x + "\n" for x in sets[p]).encode()).hexdigest()
        assert not c.get(key + "_sha256") or sha == c[key + "_sha256"], f"{key} sha {sha} != {c[key + '_sha256']}"
        pools[p] = {"n": len(sets[p]), "sha256": sha, "path": c[key]}
    every, budget = c["eval_every_rollouts"], c["budget_rollouts"]
    assert every > 0 and every % GEN == 0 and budget >= every and budget % every == 0, \
        f"eval_every_rollouts must be a multiple of {GEN} and budget_rollouts a multiple of it"
    assert f"--model {PINNED_MODEL} --effort {EFFORT}" in OVERSEER_SH.read_text(), f"{OVERSEER_SH} does not pin {PINNED_MODEL} at {EFFORT}"
    root, clamps, err = knobs.check_config({k: knobs.ROOT[k] for k in knobs.knob_names()})
    assert root == knobs.ROOT and not clamps and not err, (root, clamps, err)
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / "root_config.json", root)  # frozen for A2; a2 asserts its sha256
    t = {"run_id": c["run_id"], "config": c, "created": time.strftime("%F %T"), "pools": pools, "n_val": len(sets["val"]),
         "knobs": {"ranges": knobs.RANGES, "fixed": knobs.FIXED, "warmup_updates": knobs.WARMUP_UPDATES},
         "root_config": root, "root_config_sha256": hashlib.sha256((d / "root_config.json").read_bytes()).hexdigest(),
         "calls": 0, "journal": [],
         "arms": {"main": {"nodes": {"n0": new_node("n0", None, dict(root), "root", 0, sampler.new(c["run_id"], "root"))},
                           "order": ["n0"], "evals": {}, "eval_order": [], "stage": {"name": "eval0", "node": "n0", "seg_end": 0},
                           "rollouts_used": 0, "budget": budget}}}
    write_json(d / "tree.json", t)
    print("INIT", d, "easy", pools["easy"]["n"], "hard", pools["hard"]["n"], "val", len(sets["val"]), "budget", budget, "eval every", every)


def status(run):
    t = run.t
    evs = run.events()
    shas = sorted({e["trainer_sha256"] for e in evs if e.get("trainer_sha256")})  # every arm's launches are in events.log
    out = {"decisions": len(t["journal"]), "fallbacks": sum(j["fallback"] for j in t["journal"]),
           "exits_after_segment": sum(e["event"] == "exit_after_segment" for e in evs), "trainer_sha256": shas,
           "warnings": [f"{GRPO_TRAINER.name} changed during this run ({len(shas)} sha256 values in the launch events): "
                        "its launches did not all train with the same loss code"] if len(shas) > 1 else []}
    if any(e["event"] == "a2_warning" for e in evs):
        out["warnings"].append("the overseer arm never left n0, so A2 seed 1 is a copy of it (see the a2_warning event)")
    for arm, A in t["arms"].items():
        out[arm] = {"stage": A["stage"], "rollouts_used": A["rollouts_used"], "budget": A["budget"],
                    "nodes": {n: {k: A["nodes"][n][k] for k in ("from", "status", "gens")} for n in A["order"]},
                    "evals": len(A["eval_order"]), "final_pick": final_pick(A)}
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "run", "a2", "status", "supervise"])
    ap.add_argument("run_dir")
    ap.add_argument("--config")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--fake", action="store_true")  # CPU tests: fake_train.py in place of the GPU launch
    ap.add_argument("--halt-before", nargs="*", default=[])
    ap.add_argument("--max-launches", type=int, default=0)
    ap.add_argument("--seed", type=int, choices=[1, 2])
    a = ap.parse_args()
    FAKE["on"] = a.fake
    if a.cmd == "init":
        return cmd_init(a.run_dir, a.config)
    if a.cmd == "supervise":
        return supervise(a.run_dir, (["--fake"] if a.fake else []) + (["--max-launches", str(a.max_launches)] if a.max_launches else [])
                         + (["--halt-before", *a.halt_before] if a.halt_before else []), a.seed)
    assert a.cmd != "a2" or a.seed, "a2 needs --seed 1 or 2"
    run = Run(a.run_dir)
    if a.cmd == "status":
        return status(run)
    sync_journal(run)
    arm = "main"
    if a.cmd == "a2":
        hold = run.d / "A2_HOLD"
        if hold.exists() and str(a.seed) in hold.read_text().split():
            run.event("a2_held", seed=a.seed, hold=hold.read_text().strip())
            print(f"A2 HELD: seed {a.seed} is listed in {hold}", flush=True)
            return EXIT_A2_HELD
        arm = a2_start(run, a.seed)
        if arm is None:
            return EXIT_A2_GATE
    launches = 0
    while True:
        stage = run.t["arms"][arm]["stage"]["name"]
        if a.max_launches and stage in ("eval0", "train", "eval") and launches >= a.max_launches:
            run.event("halt", why=f"max launches {a.max_launches}")
            return 0
        launches += stage in ("eval0", "train", "eval")
        rc = step_stage(run, a.go, a.halt_before, arm)
        if rc is not None:
            return rc
        if not a.go:
            return 0


if __name__ == "__main__":
    sys.exit(main())
