#!/usr/bin/env bash
# Page-cache eviction sidecar for GPU runs (added after a host-RAM kill at startup).
# WSL's page cache counts as used on Windows, and watch.sh's 4 GB host-RAM guard reads Windows free. evict.py drops
# the clean cache of our own files (posix_fadvise DONTNEED, no root): the HF blobs and vLLM's and Triton's compile
# caches. Pages a live process maps (the venv's libraries) stay. [measured] Dropping 3.3 GB took about 20 s,
# and Windows free rose about 10 s after that.
# Usage (in WSL, started by run/orchestrator.py): evict_sidecar.sh <run log> until_step|whole_run
#   until_step: an eviction pass every 5 s until the run's first STEP line. whole_run: every 5 s for 10 min, then
#   every 60 s. In the first 3 min, also a blob-only pass every 1 s (added after a second startup kill). Both
#   stop at EXIT or WATCHDOG in the run log (checked every 5 s), or if the log never appears.
#   One line per pass in <run>.evict.
# Spill diagnostics (added after two spill pauses in a calibration run; the guard itself stays in watch.sh):
#   every 60 s, one "gpu" line from a single Get-Counter call (scripts/gpu_trace.ps1): each adapter's dedicated and
#   shared GB (the counters watch.sh reads), so a gradual rise shows apart from a jump, plus host process objects
#   against live processes, paged and nonpaged pool, and available MB (for the host-RAM gap). Then one
#   "gpu_procs" line with the top 5 processes by shared GB on each adapter (scripts/gpu_procs.ps1), which says who
#   holds the shared bytes. On a WATCHDOG line in the run log, a
#   gpu_procs line and Windows nvidia-smi's full output before stopping. Every poll has a 25 s timeout, as in
#   watch.sh; a hung poll is logged. GPU_PS, PROC_PS, NVSMI and BLOBS override the pollers and the blob dir (tests only).
L=$1; mode=$2; out=${L%.log}.evict; t0=$(date +%s); tg=0; te=0
HERE=$(cd "$(dirname "$0")" && pwd)
GPU_PS=${GPU_PS:-$(wslpath -w $HERE/gpu_trace.ps1)}
PROC_PS=${PROC_PS:-$(wslpath -w $HERE/gpu_procs.ps1)}
NVSMI=${NVSMI:-/mnt/c/Windows/System32/nvidia-smi.exe}
BLOBS=${BLOBS:-$HOME/llm-overseer-rl/hf/hub/blobs}
# + the run dir (log's grandparent): the main run's checkpoints grew the cache 1.6 GB per segment, killing the eval at 3.9 GB free
DIRS="$BLOBS $HOME/.cache/vllm $HOME/.triton $(dirname "$(dirname "$L")")"
WIN='[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)'
cache() { free -m | awk '/^Mem/{print $6}'; }
pspoll() { timeout 25 powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$1" 2>/dev/null | tr -d '\r' | sort | tr '\n' ';'; }
procs() { local p; p=$(pspoll "$PROC_PS"); echo "$(date +%T) gpu_procs top5 per adapter, pid name shared_GB: ${p:-poll failed, timed out or empty}" >> "$out"; }
echo "$(date +%T) start $mode, log $L" >> "$out"
while :; do
  now=$(date +%s); age=$(( now - t0 )); why=
  grep -qE '^EXIT|WATCHDOG' "$L" 2>/dev/null && why=exit
  [ "$mode" = until_step ] && grep -q '^STEP ' "$L" 2>/dev/null && why=first_step
  [ -e "$L" ] || [ $age -lt 120 ] || why=no_log  # watch.sh exits 98 before creating the log if its GPU poll fails
  if [ -n "$why" ]; then
    if grep -q WATCHDOG "$L" 2>/dev/null; then
      echo "$(date +%T) guard alarm in the run log: $(grep -m1 WATCHDOG "$L")" >> "$out"
      procs
      smi=$(timeout 25 $NVSMI 2>&1); rc=$?
      { echo "$(date +%T) nvidia-smi (exit $rc, 124 = timed out):"; echo "$smi" | tr -d '\r'; } >> "$out"
    fi
    echo "$(date +%T) stop: $why" >> "$out"; exit 0
  fi
  if [ $(( now - tg )) -ge 60 ]; then
    tg=$now
    g=$(pspoll "$GPU_PS")
    echo "$(date +%T) gpu luid dedicated_GB shared_GB, host: ${g:-poll failed or timed out}" >> "$out"
    procs
  fi
  if [ "$mode" != whole_run ] || [ $age -lt 600 ] || [ $(( now - te )) -ge 60 ]; then
    te=$now
    win=$(timeout 25 powershell.exe -NoProfile -Command "$WIN" 2>/dev/null | tr -d '\r')
    c0=$(cache); s=$(date +%s%N)
    /usr/bin/python3 $HERE/evict.py $DIRS >/dev/null 2>&1
    echo "$(date +%T) windows_free_GB=$win wsl_cache_MB=$c0 after_evict_MB=$(cache) evict_ms=$(( ($(date +%s%N) - s) / 1000000 ))" >> "$out"
  fi
  if [ $age -lt 180 ]; then
    # fast blob pass every 1 s for the first 3 min: vLLM maps the 3.4 GB blob while it loads, and its pages can only
    # be dropped once it unmaps them (one pass kept 5,052 MB, the pass 7 s later dropped to 2,241, the guard polled 1 s after that)
    for k in 1 2 3 4 5; do
      b=$(cache); /usr/bin/python3 $HERE/evict.py $BLOBS >/dev/null 2>&1; a=$(cache)
      [ $(( b - a )) -ge 100 ] && echo "$(date +%T) fast_evict blobs wsl_cache_MB=$b after_MB=$a" >> "$out"
      sleep 1
    done
  else sleep 5; fi
done
