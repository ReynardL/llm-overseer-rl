# Source from WSL: . scripts/env.sh
# The venv and the HF cache live on WSL's ext4 disk, under ~/llm-overseer-rl.
export CRL=~/llm-overseer-rl
export HF_HOME=$CRL/hf
export PATH=$CRL/venvs/unsloth/bin:$PATH
export REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TRL_EXPERIMENTAL_SILENCE=1
# No nvcc in WSL (no CUDA toolkit, needs sudo); FlashInfer JIT-compiles its sampler, so use vLLM's PyTorch sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0
export UNSLOTH_VLLM_STANDBY=1 UNSLOTH_VLLM_STANDBY_UTIL_OVERRIDE=1 UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD=1
# No sequence packing. run/trainer.py asserts this and the line above.
export UNSLOTH_GRPO_SEQ_PACKING=0
export HF_HUB_OFFLINE=1
