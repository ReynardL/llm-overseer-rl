#!/usr/bin/env bash
# CPU/network only. Downloads Qwen3 weights into $HF_HOME.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export HF_HUB_OFFLINE=0  # env.sh turns the hub off for runs
for m in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B; do
  hf download "$m"
done
du -sh "$HF_HOME/hub"/*
