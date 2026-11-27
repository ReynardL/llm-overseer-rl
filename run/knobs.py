"""Knobs: the root config, allowed ranges, the width limit, the decision schema, its validator and the
GRPOConfig mapping.

One source for the trainer (trainer.py --node-config), the orchestrator's prompt and schema, and its validator.
Stdlib only.  python3 knobs.py   self-check
"""
import copy
import json

GEN_ROLLOUTS = 32  # fixed: rollouts per generation. Prompts per generation P = 32 / G, updates per generation U = 32 / rollouts_per_update
WARMUP_UPDATES = 10  # [default] optimizer steps of linear LR warmup on a fresh optimizer (constant_with_warmup)

# Root config. reset_optimizer True at the root: it has no parent, so its optimizer is fresh.
ROOT = {"learning_rate": 2e-5, "num_generations": 4, "rollouts_per_update": 16, "hard_share": 0.0, "reset_optimizer": True,
        "scale_rewards": "batch", "temperature": 1.0, "max_completion_length": 3472, "max_grad_norm": 1.0, "beta": 0.0}

# Allowed ranges. Also the JSON schema. "group" scaling is removed (with partial credit it would blow a small gap up).
RANGES = {
    "learning_rate": {"type": "number", "minimum": 1e-6, "maximum": 1e-4},
    "num_generations": {"type": "integer", "enum": [4, 8]},
    "rollouts_per_update": {"type": "integer", "enum": [16, 32]},
    "hard_share": {"type": "number", "enum": [0.0, 0.25, 0.5, 0.75, 1.0]},  # 0.75 and 1.0 were added mid-run, at e57
    "reset_optimizer": {"type": "boolean"},
    "scale_rewards": {"type": "string", "enum": ["batch", "none"]},
    "temperature": {"type": "number", "minimum": 0.6, "maximum": 1.2},
    "max_completion_length": {"type": "integer", "minimum": 1024, "maximum": 3472},  # 3472 = 4096 - 624-token max prompt
    "max_grad_norm": {"type": "number", "minimum": 0.1, "maximum": 2.0},
}

# VRAM guard: micro-batch 1, so training memory follows one trajectory's width; vLLM's memory is preallocated.
PROMPT_MAX = 624
WIDTH_LIMIT = 4096  # PROMPT_MAX + max_completion_length must not exceed it
VRAM_CAP_GIB = 7.92
VRAM_BAR_MIB = 300
MICRO_BATCH = 1

# Set explicitly for every node and A2. The LR schedule is not here: it is set per node in trainer_kwargs.
FIXED = {"loss_type": "dapo", "importance_sampling_level": "token", "num_iterations": 1,
         "top_p": 1.0, "top_k": 0, "min_p": None, "repetition_penalty": 1.0,  # top_k 0 = off (Unsloth passes -1 to vLLM)
         "mask_truncated_completions": True, "max_tool_calling_iterations": 10,
         "vllm_importance_sampling_correction": True, "vllm_importance_sampling_mode": "sequence_mask",
         "vllm_importance_sampling_clip_max": 3.0,
         "optim": "adamw_8bit", "weight_decay": 0.001,
         "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-8, "shuffle_dataset": False,
         "per_device_train_batch_size": MICRO_BATCH,
         "epsilon": 0.2, "epsilon_high": 0.2}

DECISIONS = ["CONTINUE", "NEW_PHASE", "REVERT"]
TEXT_FIELDS = ["reason", "prediction", "change_mind", "grade_previous"]


def knob_names():
    return list(RANGES)


def prompts_per_gen(c):
    return GEN_ROLLOUTS // c["num_generations"]


def updates_per_gen(c):
    return GEN_ROLLOUTS // c["rollouts_per_update"]


def trainer_kwargs(c):
    """GRPOConfig kwargs for config c (all knobs plus FIXED)."""
    assert GEN_ROLLOUTS % c["num_generations"] == 0 and GEN_ROLLOUTS % c["rollouts_per_update"] == 0, c
    assert PROMPT_MAX + c["max_completion_length"] <= WIDTH_LIMIT, c
    fresh = c["reset_optimizer"]
    return {**FIXED, "learning_rate": c["learning_rate"], "beta": c["beta"], "temperature": c["temperature"],
            "max_completion_length": c["max_completion_length"], "num_generations": c["num_generations"],
            "steps_per_generation": GEN_ROLLOUTS // MICRO_BATCH,
            "gradient_accumulation_steps": c["rollouts_per_update"] // MICRO_BATCH,
            "max_grad_norm": c["max_grad_norm"], "scale_rewards": c["scale_rewards"],
            "lr_scheduler_type": "constant_with_warmup" if fresh else "constant",
            "warmup_steps": WARMUP_UPDATES if fresh else 0}


def config_schema():
    return {"type": "object", "properties": {k: RANGES[k] for k in knob_names()}, "required": knob_names(),
            "additionalProperties": False}


def decision_schema(targets):
    """The overseer's answer. target and config are read for REVERT (both) and NEW_PHASE (config); CONTINUE ignores them."""
    s = {"type": "string"}
    return {"type": "object", "properties": {
        "decision": {"type": "string", "enum": DECISIONS},
        "target": {"type": "string", "enum": list(targets)},
        "config": config_schema(), **{k: s for k in TEXT_FIELDS}},
        "required": ["decision"] + TEXT_FIELDS, "additionalProperties": False}


def ranges_text():
    lines = []
    for k in knob_names():
        r = RANGES[k]
        lines.append(f"- {k}: " + (f"one of {r['enum']}" if "enum" in r else "true or false" if r["type"] == "boolean"
                                   else f"{r['type']} from {r['minimum']:g} to {r['maximum']:g}"))
    lines.append(f"- a generation is always {GEN_ROLLOUTS} rollouts: prompts per generation = {GEN_ROLLOUTS} / num_generations, "
                 f"optimizer updates per generation = {GEN_ROLLOUTS} / rollouts_per_update.")
    lines.append("- beta is fixed at 0.0 (no KL term) and the clip range (epsilon) is fixed at "
                 f"{FIXED['epsilon']:g}.")
    return "\n".join(lines)


def check_config(c):
    """(full config, clamps, error). Out-of-range numbers are clamped; wrong keys, types or enums are errors."""
    clamps = []
    if not isinstance(c, dict) or set(c) != set(knob_names()):
        return None, clamps, f"config keys must be exactly {knob_names()}; got {sorted(c) if isinstance(c, dict) else c!r}"
    c = dict(c)
    for k in knob_names():
        r, v = RANGES[k], c[k]
        ok_type = {"number": (int, float), "integer": (int,), "string": (str,), "boolean": (bool,)}[r["type"]]
        if (isinstance(v, bool) and r["type"] != "boolean") or not isinstance(v, ok_type):
            return None, clamps, f"{k}={v!r} must be a {r['type']}"
        if "enum" in r:
            if v not in r["enum"]:
                return None, clamps, f"{k}={v!r} is not one of {r['enum']}"
        elif "minimum" in r and not r["minimum"] <= v <= r["maximum"]:
            nv = min(max(v, r["minimum"]), r["maximum"])
            clamps.append({"knob": k, "from": v, "to": nv, "why": "outside the allowed range"})
            c[k] = nv
    c["hard_share"] = float(c["hard_share"])
    c["beta"] = 0.0
    assert PROMPT_MAX + c["max_completion_length"] <= WIDTH_LIMIT, c  # holds by construction: 624 + at most 3472
    return c, clamps, None


def check_decision(p, targets, base):
    """Validate an overseer answer. Returns (decision, clamps, error); error is None when usable.
    targets: the kept eval point ids a REVERT may name; base: the base eval's id (it has no optimizer to keep).
    The returned decision has decision, the four text fields, and for NEW_PHASE/REVERT a full config (REVERT: target)."""
    clamps = []
    if not isinstance(p, dict) or p.get("decision") not in DECISIONS:
        return None, clamps, f"decision must be one of {DECISIONS}; got {p.get('decision') if isinstance(p, dict) else type(p).__name__!r}"
    extra = set(p) - {"decision", "target", "config", *TEXT_FIELDS}
    if extra:
        return None, clamps, f"unknown keys {sorted(extra)}"
    for k in TEXT_FIELDS:
        if not isinstance(p.get(k), str):
            return None, clamps, f"{k} must be a string"
    d = {"decision": p["decision"], **{k: p[k] for k in TEXT_FIELDS}}
    if d["decision"] == "CONTINUE":
        return d, clamps, None
    if d["decision"] == "REVERT":
        if p.get("target") not in targets:
            return None, clamps, f"REVERT target {p.get('target')!r} is not one of the kept eval points {list(targets)}"
        d["target"] = p["target"]
    if "config" not in p:
        return None, clamps, f"{d['decision']} needs a config with every knob"
    c, clamps, err = check_config(p["config"])
    if err:
        return None, clamps, err
    if d.get("target") == base and not c["reset_optimizer"]:
        clamps.append({"knob": "reset_optimizer", "from": False, "to": True, "why": "the base model has no optimizer state to keep"})
        c["reset_optimizer"] = True
    d["config"] = c
    return d, clamps, None


if __name__ == "__main__":
    kw = trainer_kwargs(ROOT)
    assert kw["steps_per_generation"] == 32 and kw["gradient_accumulation_steps"] == 16 and kw["per_device_train_batch_size"] == 1
    assert kw["lr_scheduler_type"] == "constant_with_warmup" and kw["warmup_steps"] == WARMUP_UPDATES and kw["learning_rate"] == 2e-5
    assert kw["scale_rewards"] == "batch" and kw["epsilon"] == kw["epsilon_high"] == 0.2 and kw["beta"] == 0.0
    assert prompts_per_gen(ROOT) == 8 and updates_per_gen(ROOT) == 2 and PROMPT_MAX + ROOT["max_completion_length"] == WIDTH_LIMIT
    kept = trainer_kwargs({**ROOT, "reset_optimizer": False, "num_generations": 8, "rollouts_per_update": 32})
    assert kept["lr_scheduler_type"] == "constant" and kept["warmup_steps"] == 0 and kept["gradient_accumulation_steps"] == 32
    assert "group" not in json.dumps(RANGES) and "beta" not in knob_names() and "epsilon" not in knob_names()
    cfg = {k: ROOT[k] for k in knob_names()}
    txt = {k: "x" for k in TEXT_FIELDS}
    T = ["e0", "e1", "e2"]
    assert check_decision({"decision": "CONTINUE", **txt}, T, "e0") == ({"decision": "CONTINUE", **txt}, [], None)
    # CONTINUE ignores a config and a target, even broken ones
    assert check_decision({"decision": "CONTINUE", "config": {"x": 1}, "target": "e9", **txt}, T, "e0")[2] is None
    d, cl, err = check_decision({"decision": "NEW_PHASE", "config": cfg, **txt}, T, "e0")
    assert err is None and d["config"] == ROOT and "target" not in d and cl == [], (d, cl, err)
    big = copy.deepcopy(cfg); big.update(learning_rate=1e-3, num_generations=8, rollouts_per_update=32, hard_share=0.5, reset_optimizer=False)
    d, cl, err = check_decision({"decision": "REVERT", "target": "e1", "config": big, **txt}, T, "e0")
    assert err is None and d["target"] == "e1" and d["config"]["learning_rate"] == 1e-4 and [x["knob"] for x in cl] == ["learning_rate"]
    assert d["config"]["reset_optimizer"] is False and prompts_per_gen(d["config"]) == 4 and updates_per_gen(d["config"]) == 1
    d, cl, err = check_decision({"decision": "REVERT", "target": "e0", "config": big, **txt}, T, "e0")
    assert err is None and d["config"]["reset_optimizer"] is True and [x["knob"] for x in cl] == ["learning_rate", "reset_optimizer"]
    d, cl, err = check_decision({"decision": "NEW_PHASE", "config": {**cfg, "hard_share": 0}, **txt}, T, "e0")
    assert err is None and d["config"]["hard_share"] == 0.0  # integer 0 is a number
    for bad, msg in [({"decision": "STOP"}, "decision must be"), ({"decision": "REVERT", "target": "e9", "config": cfg}, "REVERT target"),
                     ({"decision": "REVERT", "config": cfg}, "REVERT target"), ({"decision": "NEW_PHASE"}, "needs a config"),
                     ({"decision": "NEW_PHASE", "config": {**cfg, "num_generations": 6}}, "num_generations"),
                     ({"decision": "NEW_PHASE", "config": {**cfg, "scale_rewards": "group"}}, "scale_rewards"),
                     ({"decision": "NEW_PHASE", "config": {**cfg, "hard_share": 0.3}}, "hard_share"),
                     ({"decision": "NEW_PHASE", "config": {**cfg, "reset_optimizer": 1}}, "reset_optimizer"),
                     ({"decision": "NEW_PHASE", "config": {**cfg, "temperature": True}}, "temperature"),
                     ({"decision": "NEW_PHASE", "config": {**cfg, "beta": 0.0}}, "config keys"),
                     ({"decision": "NEW_PHASE", "config": {k: v for k, v in cfg.items() if k != "temperature"}}, "config keys"),
                     ({"decision": "CONTINUE", "reason": 3}, "reason must be"), ({"decision": "CONTINUE", "rationale": "x"}, "unknown keys")]:
        got = check_decision({**txt, **bad}, T, "e0")
        assert got[0] is None and msg in got[2], (bad, got)
    assert check_decision("CONTINUE", T, "e0")[2] and check_decision(None, T, "e0")[2]
    s = decision_schema(T)
    assert s["properties"]["target"]["enum"] == T and s["properties"]["config"]["required"] == knob_names() and json.dumps(s)
    assert "config" not in s["required"] and "target" not in s["required"]
    print("KNOBS_CHECK ok\n" + ranges_text())
