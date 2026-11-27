"""Overseer prompt: stateless, one text per decision, built from tree.json alone.
Val reaches it as aggregates only (pass rate, SE, paired deltas); training figures come from training rollouts only.
No trajectories, no task ids, and no figure measured in an earlier phase."""
import knobs
import sampler
import trainenv  # CAP only

PROMPT_VERSION = "p5-2026-11-21"  # bump on any prompt text change; logged in every overseer request
FULL_ENTRIES = 10  # a compacted journal keeps this many newest entries in full

TASK = ("You oversee one long reinforcement-learning run (GRPO) of a 1.7B code model (Qwen3-1.7B with a LoRA adapter). "
        "The task is multi-turn bug repair: the model gets a buggy Python function, uses the tools view_file, edit_file, "
        "run_tests and submit for up to 10 turns, and is graded on hidden tests.\n"
        "Eval reward is binary: 1 only if every hidden test passes. After every {every} training rollouts the current "
        "checkpoint is scored greedily on a fixed set of {n_val} val tasks; that is one eval point.\n"
        "Training reward: 1.0 for a full pass. A submission that loads and runs but fails some hidden tests earns "
        "{cap:g} times the share of hidden tests it fixes beyond those the unedited buggy file already passes. No "
        "submission, a load error, a timeout or an out-of-memory grade earn 0.\n"
        "Training tasks come from two pools, both disjoint from val: easy ({easy} tasks the base model solved 1 to 6 times "
        "in 8 calibration samples) and hard ({hard} tasks it solved 0 times in 8). Val was not filtered by difficulty.")

GOAL = ("Goal: the val pass rate of the final pick. After the run, the eval point with the highest mean val pass rate "
        "over itself and the two evals before it on its path is picked (ties go to the earliest), so a single high eval "
        "counts less than a sustained level. The budget is {budget} training rollouts and is always spent in full: there "
        "is no early stop. Used {used}, remaining {left}. Evals remaining: {evals_left}. This is decision {i} of {n}.")

MECHANICS = (
    "Mechanics:\n"
    "- A generation is always {gen} rollouts sampled from the current weights. num_generations (G) is the rollouts per "
    "prompt, so a generation has {gen}/G prompts. Advantages are computed once per generation, over all {gen} rollouts.\n"
    "- rollouts_per_update: 16 gives two optimizer updates per generation, 32 gives one. With two, the second update "
    "trains on rollouts sampled before the first update. The loss uses the sampler's own log-probabilities as the old "
    "policy, so the second update has a true PPO ratio against the sampling policy, clipped at {eps:g}. The clip ratio "
    "of a first update shows only numeric mismatch between trainer and sampler; of a second update it also shows how "
    "far the first update moved the policy.\n"
    "- hard_share: the share of each generation's prompts drawn from the hard pool (rounded to whole prompts), the "
    "rest from the easy pool.\n"
    "- scale_rewards: \"batch\" divides advantages by the standard deviation of the reward over the generation's {gen} "
    "rollouts; \"none\" leaves them unscaled.\n"
    "- reset_optimizer (read for NEW_PHASE and REVERT): true starts the phase with a fresh optimizer and a linear "
    "learning-rate warmup from 0 over its first {warm} updates. false keeps the optimizer state of the checkpoint the "
    "phase starts from, with the new learning rate and no warmup. A start from the base model always has a fresh optimizer.\n"
    "- Zero-signal rule (fixed, the same in every run, outside your control): a visit to a task is zero-signal when all "
    "G rollouts of its prompt got the same training reward. {streak} zero-signal visits in a row rest the task for "
    "{rest} rollouts of its lineage; it returns with a clean count. The rule is applied at eval boundaries.\n"
    "- Decisions. CONTINUE: the same phase trains the next segment; nothing is reset; config and target are ignored. "
    "NEW_PHASE: a new phase starts from the latest checkpoint with the full config you give (the current config with "
    "reset_optimizer true is a pure optimizer reset). REVERT: a new phase starts from the checkpoint of an earlier eval "
    "point (target: any id in the eval table except the latest, the base e0 included) with the full config you give. A revert is not "
    "free: the rollouts spent on the abandoned branch still count against the budget.\n"
    "- Every eval point keeps its checkpoint and its task-sampler state (pool positions, zero-signal counts, rests). A "
    "new phase or a revert continues from the state stored with its start point.\n"
    "- Val noise: each pass rate has a standard error from a cluster bootstrap over source functions (+-SE). Every "
    "eval scores the same val tasks, so the difference between two evals has its own paired SE, shown with each delta. "
    "Judge differences by their paired SE.\n"
    "- Training figures are not comparable with val: training samples at the phase's temperature from the pools above "
    "and pays partial credit; val is greedy, unfiltered and binary.")

LEGEND = ("Training summary per segment, from training rollouts only. reward = mean training reward; pass = share with a "
          "full pass; partial = share paid partial credit; no-submit = share that never submitted; wrong = share that "
          "submitted code that loaded but did not pass every hidden test (partial is inside it); load-err = share whose "
          "submission failed to load; trunc = share cut at the completion cap (masked from the loss); tools = mean tool "
          "calls per rollout; zero-groups = share of prompts whose G rollouts all got the same reward; tokens = mean "
          "completion tokens; pools = active/rested tasks after the segment; u1 and u2 = means over the first and second "
          "update of each generation (loss, grad norm before clipping, clip ratio); lr = learning rate after the "
          "segment's last update.")

ANSWER = ("Answer with: decision (CONTINUE, NEW_PHASE or REVERT); target (REVERT only: an eval id); config (NEW_PHASE and "
          "REVERT: every knob); reason; prediction (what the next eval should show); change_mind (what result would make "
          "you change course); grade_previous (how your last prediction held up against the result that followed; an "
          "empty string on the first decision). Keep each text field under 100 words.")


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def fmt(v):
    if v is None:
        return "-"
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def tag(config):
    """A config as its differences from the root config."""
    return ", ".join(f"{k}={fmt(config[k])}" for k in knobs.knob_names() if config[k] != knobs.ROOT[k]) or "root config"


def delta(d):
    return "-" if not d else f"{100 * d['delta']:+.1f}+-{pct(d['se'])}"


def eval_line(arm, e):
    return (f"{e['id']} | {e['node']} | {e['prev'] or '-'} | {e['lineage_rollouts']} | {pct(e['val']['pass_rate'])}+-{pct(e['val']['se'])}"
            f" | {delta(e.get('vs_base'))} | {delta(e.get('vs_prev'))} | {pct(e.get('trail3'))}")


def train_line(arm, e):
    s, n = e["train"], arm["nodes"][e["node"]]
    P = knobs.prompts_per_gen(n["config"])
    pools = e["pools"]
    upd = [f"{k} loss {fmt(u['loss'])} grad {fmt(u['grad_norm'])} clip {fmt(u['clip'])}"
           + (f" entropy {fmt(u['entropy'])}" if u.get("entropy") is not None else "")
           for k, u in (("u1", s["updates"]["first"]), ("u2", s["updates"]["second"])) if u]
    return (f"{e['id']} ({e['node']}, generations {e['gen'] - s['rollouts'] // knobs.GEN_ROLLOUTS}-{e['gen']}, {P} prompts x "
            f"{n['config']['num_generations']}): reward {fmt(s['reward'])}, pass {pct(s['pass'])}%, partial {pct(s['partial'])}%, "
            f"no-submit {pct(s['no_submit'])}%, wrong {pct(s['wrong'])}%, load-err {pct(s['load_error'])}%, trunc {pct(s['trunc'])}%, "
            f"tools {fmt(s['tool_calls'])}, zero-groups {pct(s['zero_groups'])}%, tokens {fmt(s['len'])}; "
            + "; ".join(f"{p} pass {pct(s['pass_by_pool'][p])}% of {s['rollouts_by_pool'][p]}, pool {pools[p]['active']}/{pools[p]['rested']}"
                        for p in sampler.POOLS)
            + "; " + "; ".join(upd) + f"; lr {fmt(s['lr_end'])}")


def journal_entry(arm, j, full):
    nxt = next((arm["evals"][e] for e in arm["eval_order"] if arm["evals"][e].get("after_decision") == j["index"]), None)
    then = (f"then {nxt['id']}: val {pct(nxt['val']['pass_rate'])}+-{pct(nxt['val']['se'])}, d vs prev {delta(nxt.get('vs_prev'))}"
            if nxt else "then: no eval yet")
    ch = ", ".join(f"{k} {fmt(a)} -> {fmt(b)}" for k, (a, b) in j["changed"].items())
    head = (f"#{j['index']} at {j['eval']} ({j['rollouts_used']} rollouts used): {j['decision']}"
            + (f" to {j['target']}" if j.get("target") else "") + (" [fallback: no valid answer, the run continued]" if j["fallback"] else "")
            + (f"; config changes: {ch or 'none'}; reset_optimizer {str(j['config']['reset_optimizer']).lower()}"
               if j["decision"] != "CONTINUE" else ""))
    if not full:
        return f"{head}; {then}"
    return "\n".join([head] + [f"  {k}: {j[k]}" for k in knobs.TEXT_FIELDS] + [f"  {then}"])


def journal_text(t, arm, limit):
    js = t["journal"]
    if not js:
        return "Journal: empty. This is your first decision."
    text = "\n".join(journal_entry(arm, j, True) for j in js)
    # the journal alone threatens the limit: older entries shrink to one line each. ~2 chars per token, measured on
    # call 45 (113,594 chars = 60,057 tokens); the old chars/3 estimate never fired and that call failed the token check.
    if len(text) // 2 > limit // 2:
        text = "\n".join(journal_entry(arm, j, i >= len(js) - FULL_ENTRIES) for i, j in enumerate(js))
    return "Journal of your earlier decisions, each followed by the val result that came after it:\n" + text


def decision(t, error=None, limit=60000):
    """The decision prompt for the overseer arm's pending decision. limit: the input-token cap (journal compaction)."""
    c, arm = t["config"], t["arms"]["main"]
    every, used, budget = c["eval_every_rollouts"], arm["rollouts_used"], arm["budget"]
    tip = arm["evals"][arm["eval_order"][-1]]
    evs = [arm["evals"][e] for e in arm["eval_order"]]
    parts = [
        TASK.format(every=every, n_val=t["n_val"], cap=trainenv.CAP, easy=t["pools"]["easy"]["n"], hard=t["pools"]["hard"]["n"]),
        GOAL.format(budget=budget, used=used, left=budget - used, evals_left=(budget - used) // every, i=len(t["journal"]) + 1,
                    n=budget // every - 1),
        MECHANICS.format(gen=knobs.GEN_ROLLOUTS, eps=knobs.FIXED["epsilon"], warm=knobs.WARMUP_UPDATES, streak=sampler.ZERO_STREAK,
                         rest=sampler.REST_ROLLOUTS),
        "Knobs and allowed values:\n" + knobs.ranges_text() + "\nRoot config: "
        + ", ".join(f"{k}={fmt(v)}" for k, v in knobs.ROOT.items() if k in knobs.knob_names()),
        "Phases (config shown as its differences from the root config):\nid | starts from | config | generations trained | status\n"
        + "\n".join(f"{n['id']} | {n['from'] or 'base model'} | {tag(n['config'])} | {n['gens']} | {n['status']}"
                    for n in (arm["nodes"][x] for x in arm["order"])),
        "Eval points (val pass rate % +-SE; deltas in points with their paired SE; trail-3 = mean of this eval and the "
        "two before it on its path):\nid | phase | previous eval on its path | lineage rollouts | val | d vs base | d vs previous | trail-3\n"
        + "\n".join(eval_line(arm, e) for e in evs),
        LEGEND + "\n" + "\n".join(train_line(arm, e) for e in evs if e.get("train")),
        journal_text(t, arm, limit),
        f"The latest eval point is {tip['id']} on phase {tip['node']} ({tag(arm['nodes'][tip['node']]['config'])}). Decide what the next "
        f"{every} rollouts train.\n" + ANSWER]
    if error:
        parts.append(f"Your previous answer was rejected: {error}. Answer again within the allowed values.")
    return "\n\n".join(parts)
