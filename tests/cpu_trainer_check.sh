#!/usr/bin/env bash
# CPU dry run of run/trainer.py (no GPU: CUDA_VISIBLE_DEVICES=, plain TRL, no vLLM, --no-exec so no model output
# runs anywhere). In WSL: bash tests/cpu_trainer_check.sh
# TRL 1.15's logprob kernel has no CPU path, so each training run stops at its first use. Before that:
#   A (fresh, root shape: G 4, 2 updates per generation): effective_args.json equals knobs.trainer_kwargs (the trainer's
#     own assert) and the assigned-rows check at generation 0. With 16 % 32 != 0 plain TRL computes old logprobs inside
#     the scoring pass (on the GPU stack the vLLM rollout logprobs are pi_old), so on CPU this run stops there, before rows.jsonl.
#   A1 (fresh, G 4, 1 update per generation): rows.jsonl = generation 0's rows x G with the trainer's fields, every row's
#     env is a TrainEnv aligned with its input row (the trainer's own assert), trajectories.jsonl joins on env_index.
#   B (--init-adapter + --init-optimizer, G 8, 1 update per generation, another LR): a saved LoRA with random B loads
#     exactly, and a saved 8-bit Adam state loads exactly with every param-group LR set to the node's.
#   N1, N2: a stream shorter than --stop-at-gen, and UNSLOTH_GRPO_TRUNCATED_SUPPORT set, both refuse to start.
# Completion cap 32 so CPU generation is quick; it bypasses the knob range on purpose.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"
export CUDA_VISIBLE_DEVICES=
SRC=$REPO/run
R=${R:-$HOME/llm-overseer-rl/runs/cpu_trainer_check}
M=Qwen/Qwen3-0.6B  # the unsloth repo's template has no TRL response schema off the Unsloth path
rm -rf $R && mkdir -p $R
python3 - <<EOF
import json, sys
sys.path.insert(0, "$SRC")
import knobs, sampler
pools = {"easy": [l.strip() for l in open("$SRC/../data/train_pool.txt") if l.strip()], "hard": [l.strip() for l in open("$SRC/../data/hard_pool.txt") if l.strip()]}
for name, cfg in (("a", dict(knobs.ROOT, max_completion_length=32)),
                  ("a1", dict(knobs.ROOT, max_completion_length=32, rollouts_per_update=32)),
                  ("b", dict(knobs.ROOT, max_completion_length=32, num_generations=8, rollouts_per_update=32, hard_share=0.5,
                             reset_optimizer=False, learning_rate=5e-6))):
    P = knobs.prompts_per_gen(cfg)
    json.dump(cfg, open(f"$R/config_{name}.json", "w"))
    json.dump({"run": "cpucheck", "name": "root", "P": P, "rows": sampler.assign(sampler.new("cpucheck", "root"), pools, P, cfg["hard_share"], 2)},
              open(f"$R/stream_{name}.json", "w"))
EOF
T="python $SRC/trainer.py --model $M --env code --tasks $CRL/tasks/v2 --no-exec --max-gens 4 --save-gens 1 --pause-file $R/PAUSE"
echo "== A fresh, root shape"; timeout 3000 $T --node-config $R/config_a.json --stream $R/stream_a.json --stop-at-gen 2 --out $R/a > $R/a.log 2>&1; echo "A exit $?"
echo "== A1 fresh, one update per generation"; timeout 3000 $T --node-config $R/config_a1.json --stream $R/stream_a1.json --stop-at-gen 2 --out $R/a1 > $R/a1.log 2>&1; echo "A1 exit $?"
grep -E "effective GRPOConfig|STREAM|AssertionError|generation [0-9]+:" $R/a.log $R/a1.log | head -5
python - <<EOF
import json, sys
sys.path.insert(0, "$SRC")
import knobs
for x in ("a", "a1"):
    eff = json.load(open(f"$R/{x}/effective_args.json"))
    want = knobs.trainer_kwargs(json.load(open(f"$R/config_{x}.json")))
    off = {k: (v, eff.get(k)) for k, v in want.items() if k != "top_k" and eff.get(k) != v}
    print(x.upper(), "effective_args", "MATCH" if not off else f"DIFF {off}", "top_k", eff["top_k"], "seed", eff["seed"],
          "spg", eff["steps_per_generation"], "accum", eff["gradient_accumulation_steps"], eff["lr_scheduler_type"], eff["warmup_steps"])
rows = [json.loads(l) for l in open("$R/a1/rows.jsonl")]
s = json.load(open("$R/stream_a1.json"))["rows"]
ok = [r["task_id"] for r in rows] == [t for t in s[:8] for _ in range(4)] and [r["row"] for r in rows] == list(range(32)) \
    and all(r["gen"] == 0 and r["step"] == 0 and r["reward"] == 0.0 and r["passed"] == 0.0 for r in rows) \
    and len({r["env_index"] for r in rows}) == 32
print("A1 rows generation 0", "MATCH" if ok else "DIFF", len(rows), "rows, keys", sorted(rows[0]))
tr = [json.loads(l) for l in open("$R/a1/trajectories.jsonl")]
print("A1 trajectories", len(tr), "records, join on env_index", "OK" if {(t["index"], t["task_id"]) for t in tr} == {(r["env_index"], r["task_id"]) for r in rows} else "DIFF",
      "passed/reward fields", sorted({(t["passed"], t["reward"]) for t in tr}))
EOF
python - <<EOF
# B's parent: a LoRA like the trainer's (r16 a32 all-linear) with random B, and the optimizer the HF Trainer builds for
# it (adamw_8bit) with a random state, saved as a checkpoint dir. bitsandbytes cannot step on CPU, so the state is
# initialised by the optimizer's own init_state and then filled with random values.
import sys
sys.path.insert(0, "$SRC")
import knobs
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
m = get_peft_model(AutoModelForCausalLM.from_pretrained("$M"), LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
torch.manual_seed(1)
for n, p in m.named_parameters():
    if "lora_B" in n:
        p.data.normal_(0, 0.01)
m.save_pretrained("$R/parent/checkpoint-20")
t = Trainer(model=m, args=TrainingArguments(output_dir="$R/parent/tmp", use_cpu=True, report_to="none", learning_rate=2e-5,
            **{k: knobs.FIXED[k] for k in ("optim", "weight_decay", "adam_beta1", "adam_beta2", "adam_epsilon")}))
opt = t.create_optimizer()
n = 0
for gi, g in enumerate(opt.param_groups):
    for pi, p in enumerate(g["params"]):
        opt.init_state(g, p, gi, pi)
        st = opt.state[p]
        st["step"] = 7
        for k, v in st.items():
            if torch.is_tensor(v) and k.startswith(("state", "absmax")):
                v.copy_(torch.randint(0, 255, v.shape, dtype=torch.uint8) if v.dtype == torch.uint8 else torch.rand(v.shape))
                n += 1
torch.save(opt.state_dict(), "$R/parent/checkpoint-20/optimizer.pt")
print("B parent optimizer", type(opt).__name__, "groups", [len(g["params"]) for g in opt.param_groups], "random state tensors", n)
EOF
echo "== B init-adapter + init-optimizer"
timeout 3000 $T --node-config $R/config_b.json --stream $R/stream_b.json --stop-at-gen 2 --out $R/b --init-adapter $R/parent/checkpoint-20 \
  --init-optimizer $R/parent/checkpoint-20 --seed 3 > $R/b.log 2>&1; echo "B exit $?"
grep -E "INIT_ADAPTER|INIT_OPTIMIZER|AssertionError" $R/b.log | cut -c1-700 | head -4
python - <<EOF
import json
d = json.load(open("$R/b/init_check.json"))
print("B init_check", "EXACT" if d["max_abs_diff"] == 0.0 and not (d["unexpected"] or d["live_not_in_file"] or d["file_not_in_live"]) else d, d["tensors"], "tensors")
o = json.load(open("$R/b/init_optimizer_check.json"))
print("B init_optimizer", "EXACT" if o["max_abs_diff"] == 0.0 and o["n_param_states"] == o["live_n_param_states"] > 0 else "DIFF",
      {k: o[k] for k in ("n_param_states", "n_tensors", "optimizer_steps", "group_lrs", "scheduler_base_lrs", "lr_scheduler_type", "group_diffs")})
rows = [json.loads(l) for l in open("$R/b/rows.jsonl")]
s = json.load(open("$R/stream_b.json"))["rows"]
print("B rows generation 0", "MATCH" if [r["task_id"] for r in rows] == [t for t in s[:4] for _ in range(8)] else "DIFF", len(rows), "rows")
EOF
echo "== N1 stream shorter than --stop-at-gen"; timeout 600 $T --node-config $R/config_a.json --stream $R/stream_a.json --stop-at-gen 3 --out $R/n1 > $R/n1.log 2>&1; echo "N1 exit $?"
grep -E "AssertionError" $R/n1.log | cut -c1-200 | head -1
echo "== N2 truncated support set"; UNSLOTH_GRPO_TRUNCATED_SUPPORT=on timeout 600 $T --node-config $R/config_a.json --stream $R/stream_a.json --stop-at-gen 2 --out $R/n2 > $R/n2.log 2>&1; echo "N2 exit $?"
grep -E "AssertionError" $R/n2.log | cut -c1-200 | head -1
echo "== last error lines of A, A1 and B (expected: the Triton kernel on CPU)"; for x in a a1 b; do grep -E "Error" $R/$x.log | tail -1 | cut -c1-300; done
