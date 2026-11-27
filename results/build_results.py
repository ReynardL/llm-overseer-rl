"""Build the published results from a finished Phase 5 run directory (which is not in this repo: it holds checkpoints).

  python3 results/build_results.py <run_dir>

Writes, next to this file: evals.json (every val eval of both arms, the picks, the test results), journal.jsonl (the
overseer's decisions as logged, unedited) and ../DECISIONS.md (the same decisions, readable).
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "run"))
import orchestrator as o  # noqa: E402

run = Path(sys.argv[1])
tree = json.loads((run / "tree.json").read_text())
journal = [json.loads(l) for l in (run / "journal.jsonl").read_text().splitlines() if l.strip()]
events = [json.loads(l) for l in (run / "events.log").read_text().splitlines() if l.strip()]
SEG = 800  # rollouts between val evals

arms, picks = {}, {}
for name, A in tree["arms"].items():
    picks[name] = o.final_pick(A)["id"]
    arms[name] = [{"eval": e, "rollouts": i * SEG, "node": A["evals"][e]["node"], "gen": A["evals"][e]["gen"],
                   "val": A["evals"][e]["val"]["pass_rate"], "se": A["evals"][e]["val"]["se"],
                   "trail3": A["evals"][e].get("trail3")} for i, e in enumerate(A["eval_order"])]

# a budget extension shows up in the first decision made after it
ext = [{"time": e["time"], "old": e["old"], "new": e["new"],
        "before_decision": next((j["index"] for j in journal if j["time"] > e["time"]), None)}
       for e in events if e.get("event") == "budget_extended"]

test = {}
for f in sorted((run / "test").glob("*.json")):
    t = json.loads(f.read_text())
    test[f.stem] = {k: t[k] for k in ("arm", "eval", "n", "functions", "passed", "pass_rate", "se")}
    if t.get("vs_base"):
        test[f.stem]["vs_base"] = {k: t["vs_base"][k] for k in ("delta", "se")}
both = o.paired(run / "test" / f"a2s1_{picks['a2s1']}" / "traj.jsonl", run / "test" / f"main_{picks['main']}" / "traj.jsonl")
test["main_minus_a2s1"] = {"delta": both["delta"], "se": both["se"]}

out = {"segment_rollouts": SEG, "n_val": tree["n_val"], "root_config": tree["root_config"], "arms": arms, "picks": picks,
       "budget_extensions": ext, "stopped_at": tree["arms"]["main"]["rollouts_used"], "test": test,
       "overseer": {"calls": len(journal), "cost_usd": round(sum(j["overseer"]["cost_usd"] for j in journal), 2),
                    "models": sorted({m for j in journal for m in j["overseer"]["models"]}),
                    "effort": sorted({j["overseer"]["effort"] for j in journal})}}
(HERE / "evals.json").write_text(json.dumps(out, indent=1) + "\n")
text = (run / "journal.jsonl").read_text()
assert not re.search(r"@|/home/|/mnt/", text), "journal holds a path or an address"
(HERE / "journal.jsonl").write_text(text)

# DECISIONS.md
val = {r["eval"]: r for r in arms["main"]}
NAMES = {"learning_rate": "learning rate", "num_generations": "group size", "rollouts_per_update": "rollouts per update",
         "hard_share": "hard-task share", "temperature": "temperature", "reset_optimizer": "reset optimizer"}


def changes(j):
    return ", ".join(f"{NAMES.get(k, k)} {a:g} to {b:g}" if isinstance(a, (int, float)) and not isinstance(a, bool)
                     else f"{NAMES.get(k, k)} {a} to {b}" for k, (a, b) in j["changed"].items()) or "none"


def what(j):
    return j["decision"] + (f" to {j['target']}" if j["target"] else "")


L = ["# Overseer decision log", "",
     f"All {len(journal)} decisions the overseer ({out['overseer']['models'][0]}) made in the main run, in order, built from "
     "`results/journal.jsonl` by `results/build_results.py`. Nothing is edited or left out.", "",
     "After every 800 training rollouts the run stops, the checkpoint is scored on the 398 validation tasks, and the overseer "
     "gets a table of training and validation metrics plus its own earlier entries. It answers with one of three decisions:", "",
     "- **CONTINUE**: train on with the same settings.",
     "- **NEW_PHASE**: train on from the current checkpoint with new settings.",
     "- **REVERT**: go back to an earlier checkpoint, with the same or new settings.", "",
     "It also writes a prediction for the next eval, what would change its mind, and a grade of its previous prediction.", "",
     "The rollout budget was 24,000 at the start. I extended it by hand three times during the run. Those points are marked "
     "below, because the overseer planned around the budget it was shown.", "",
     "## Summary", "", "| # | Rollouts | Val pass | Decision | Settings changed |", "|---|---|---|---|---|"]
for j in journal:
    for x in ext:
        if x["before_decision"] == j["index"]:
            L.append(f"| | | | *budget extended by hand* | *{x['old']:,} to {x['new']:,} rollouts* |")
    L.append(f"| [{j['index']}](#decision-{j['index']}) | {j['rollouts_used']:,} | {val[j['eval']]['val'] * 100:.1f}% | {what(j)} | {changes(j)} |")
L += ["", f"The run was stopped by hand at {out['stopped_at']:,} rollouts. The final pick is the eval with the best 3-eval mean "
      f"validation pass rate: {picks['main']}.", "", "## Full entries", ""]
for j in journal:
    for x in ext:
        if x["before_decision"] == j["index"]:
            L += [f"> Budget extended by hand from {x['old']:,} to {x['new']:,} rollouts ({x['time'][:10]}).", ""]
    c = j["config"]
    L += [f"### Decision {j['index']}", "",
          f"**{what(j)}** after eval {j['eval']} ({j['rollouts_used']:,} rollouts used, validation pass rate "
          f"{val[j['eval']]['val'] * 100:.1f}%). Settings changed: {changes(j)}.", "",
          f"Settings from here: learning rate {c['learning_rate']:g}, group size {c['num_generations']}, rollouts per update "
          f"{c['rollouts_per_update']}, hard-task share {c['hard_share']:g}, temperature {c['temperature']:g}.", "",
          f"**Reason.** {j['reason']}", "", f"**Prediction.** {j['prediction']}", "",
          f"**Would change its mind if.** {j['change_mind']}", ""]
    if j["grade_previous"]:
        L += [f"**Grade of its previous prediction.** {j['grade_previous']}", ""]
(HERE.parent / "DECISIONS.md").write_text("\n".join(L))
print("evals.json, journal.jsonl, DECISIONS.md written;", len(journal), "decisions; picks", picks, "; extensions", ext)
print(json.dumps(test))
