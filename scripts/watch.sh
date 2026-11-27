#!/usr/bin/env bash
# Run one GPU step in WSL under a watchdog, with a hard-stop spill guard.
# Usage: [SPILL_GUARD=0] [PAUSE_FILE=<WSL path>] bash watch.sh <name> <command run in the repo root inside WSL...>
# - Kills the run if host free RAM < 4 GB or its log is silent for 600 s.
# - Spill guard (on by default; SPILL_GUARD=0 only for CPU-only runs). Before the run it records each GPU adapter's
#   shared (system RAM) usage. During the run, the adapter holding over 1 GB dedicated is the 4060 (the iGPU holds
#   ~0.4 GB). An alarm is: that adapter's shared usage above its baseline + SPILL_GB (default 0.3), or 2 GPU polls
#   in a row that fail or take over 25 s (each failure is logged with its duration). One failure is not an alarm: in
#   an earlier run all WSL interop stalled once for about a minute, while idle polls take 1.6 to
#   1.8 s from both WSL and Git Bash [measured, 10 each], far under the 25 s timeout. On an alarm it touches PAUSE_FILE (PauseCallback saves after the current step and
#   exits), and kills the run if it is still alive GRACE s later (default 240; at once if PAUSE_FILE is unset).
# Log: ~/llm-overseer-rl/runs/$RUNDIR/<name>.log (RUNDIR default misc). Exit code: the run's, 97 if the watchdog killed it,
# 98 if the baseline poll failed (the run is not started).
export MSYS_NO_PATHCONV=1
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}  # the repo's WSL path; set REPO yourself when calling from Git Bash
name=$1; shift; cmd="$*"; d=${RUNDIR:-misc}
guard=${SPILL_GUARD:-1}; rise=${SPILL_GB:-0.3}; grace=${GRACE:-240}; poll=${POLL:-30}; pause=${PAUSE_FILE:-}
# cygpath from Git Bash, wslpath when the orchestrator runs it inside WSL (wsl.exe and powershell.exe work there too)
GPU_PS=${GPU_PS:-$( (command -v cygpath >/dev/null && cygpath -w "$(dirname "$0")/gpu_mem.ps1") || wslpath -w "$(cd "$(dirname "$0")" && pwd)/gpu_mem.ps1")}
R="~/llm-overseer-rl/runs/$d"; L="$R/$name.log"
say() { echo "[watch $(date +%T)] $*"; }
gpu() { timeout 25 powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$GPU_PS" 2>/dev/null | tr -d '\r'; }
# "<luid> <shared GB> <rise GB>" for the adapter with over 1 GB dedicated, or "none" if no adapter holds that much yet.
spill() { BASE="$base" NOW="$1" DED="${DED_GB:-1}" awk 'BEGIN{
  n=split(ENVIRON["BASE"],b,"\n"); for(i=1;i<=n;i++){split(b[i],f," "); B[f[1]]=f[3]}
  n=split(ENVIRON["NOW"],c,"\n"); bd=ENVIRON["DED"]+0; best=""
  for(i=1;i<=n;i++){split(c[i],f," "); if(f[2]+0>bd){bd=f[2]+0; best=f[1]; sh=f[3]+0}}
  if(best=="") print "none"; else printf "%s %.3f %.3f\n", best, sh, sh-B[best] }'; }
killrun() { say "KILL: $1"; wsl.exe -e bash -c "p=\$(cat $L.pid); kill -TERM -\$p 2>/dev/null; sleep 10; kill -KILL -\$p 2>/dev/null; echo WATCHDOG_KILL $1 >> $L"; killed=1; }

if [ "$guard" = 1 ]; then
  base=$(gpu)
  [ -n "$base" ] || { say "baseline GPU poll failed; run not started"; exit 98; }
  say "baseline: $(echo "$base" | tr '\n' ';')"
fi
# The command goes to a script file, so its quoting survives; setsid gives it a process group to kill.
printf '%s\n' "$cmd" | wsl.exe -e bash -c "mkdir -p $R; cat > $R/$name.sh"
wsl.exe -e bash -c ". $REPO/scripts/env.sh; cd $REPO; date +%T > $L; setsid bash $R/$name.sh >> $L 2>&1 & echo \$! > $L.pid; wait \$!; echo EXIT \$? >> $L; date +%T >> $L" &
pid=$!; minfree=999; maxrise=0; alarm_t=0; killed=0; fails=0
sleep 5
while kill -0 $pid 2>/dev/null; do
  free=$(timeout 25 powershell.exe -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" | tr -d '\r')
  age=$(wsl.exe -e bash -c "echo \$(( \$(date +%s) - \$(stat -c %Y $L) ))" | tr -d '\r')
  [ -n "$free" ] && awk "BEGIN{exit !($free < $minfree)}" && minfree=$free
  if [ -n "$free" ] && awk "BEGIN{exit !($free < 4)}"; then killrun "host free RAM ${free} GB"; break; fi
  if [ "${age:-0}" -gt 600 ]; then killrun "log silent ${age} s"; break; fi
  if [ "$guard" = 1 ]; then
    t0=$(date +%s); now=$(gpu); why=""
    if [ -z "$now" ]; then
      fails=$((fails + 1)); say "GPU poll failed or timed out after $(( $(date +%s) - t0 )) s ($fails in a row)"
      [ "$fails" -ge 2 ] && why="GPU poll failed or timed out $fails times in a row"
    else
      fails=0
      s=$(spill "$now")
      if [ "$s" != none ]; then
        set -- $s
        awk "BEGIN{exit !($3 > $maxrise)}" && maxrise=$3
        awk "BEGIN{exit !($3 > $rise)}" && why="spill: $1 shared $2 GB, +$3 GB over baseline"
      fi
    fi
    if [ -n "$why" ] && [ "$alarm_t" = 0 ]; then
      say "ALARM: $why"; alarm_t=$(date +%s)
      if [ -n "$pause" ]; then wsl.exe -e bash -c "touch $pause; echo WATCHDOG_PAUSE $why >> $L"; say "created $pause"
      else killrun "$why"; break; fi
    fi
  fi
  if [ "$alarm_t" != 0 ] && [ $(( $(date +%s) - alarm_t )) -ge "$grace" ]; then killrun "no exit ${grace} s after PAUSE"; break; fi
  sleep "$poll"
done
wait $pid
say "min_host_free_GB=$minfree max_spill_rise_GB=$maxrise alarm=$([ "$alarm_t" = 0 ] && echo no || echo yes) killed=$killed"
wsl.exe -e bash -c "grep -E 'STEP |SUMMARY|GATE|EVAL|Error|EXIT|WATCHDOG' $L | cut -c1-900 | tail -30; tail -1 $L"
code=$(wsl.exe -e bash -c "grep -oP '^EXIT \K\d+' $L | tail -1" | tr -d '\r')
[ "$killed" = 1 ] && exit 97
exit "${code:-1}"
