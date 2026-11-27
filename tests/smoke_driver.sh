#!/usr/bin/env bash
# GPU smoke driver (WSL): checks S1 to S6 as one orchestrated run of configs/smoke.json
# (3 generations per segment, 4 segments, 50 val tasks). Every GPU launch goes through the orchestrator's gates,
# scripts/watch.sh and the eviction sidecar. The overseer is fake_overseer.py with a script, except the one real call of S6.
#   bash smoke_driver.sh <config> <run_dir> main   (a fresh run dir)
#     run 1: base eval, then n0's first segment. The driver touches PAUSE after STEP 1, between the two updates of
#            generation 0; the trainer saves at the generation boundary and exits 3 (S2).
#     run 2: resume to the end of the segment (S1: root config, 3 generations), decision 1 CONTINUE, segment 2,
#            decision 2 NEW_PHASE (G 8, one update per generation, hard_share 0.5, optimizer kept: S3).
#     run 3: segment 3 on the new phase, halt before decision 3.
#     S6:    tree.json and root_config.json copied to <run_dir>.s6; one real overseer call there, nothing launched.
#     run 4: decision 3 REVERT to e1 with a fresh optimizer (S3), segment 4, done.
#   bash smoke_driver.sh <config> <run_dir> a2     (after main): A2 seed 1, then seed 2 (S5)
# An exit the driver did not plan stops it. Last line: DRIVER_EXIT <code>.
set -u
SRC=$(cd "$(dirname "$0")/../run" && pwd); CFG=$1; D=$2; PART=$3
orch() {  # $1 label, $2 the overseer (fake|real), then the orchestrator's arguments; sets rc
  local label=$1 who=$2; shift 2
  echo "=== $label $(date +%T): $*" | tee -a $D/orch.out
  if [ $who = real ]; then python3 -u $SRC/orchestrator.py "$@" 2>&1 | tee -a $D/orch.out; rc=${PIPESTATUS[0]}
  else OVERSEER_CMD="python3 $SRC/../tests/fake_overseer.py" FAKE_OVERSEER=$D/fake_overseer.json python3 -u $SRC/orchestrator.py "$@" 2>&1 | tee -a $D/orch.out; rc=${PIPESTATUS[0]}
  fi
  echo "=== $label exit $rc $(date +%T)" | tee -a $D/orch.out
}
stop() { echo "$1"; python3 $SRC/orchestrator.py status "$D"; echo "DRIVER_EXIT $rc"; exit $rc; }

if [ $PART = a2 ]; then
  for s in 1 2; do
    orch "a2 seed $s" fake a2 "$D" --seed $s --go
    [ $rc = 0 ] || stop "a2 seed $s did not finish: stopping"
  done
  stop "a2 done"
fi
[ $PART = main ] || { echo "usage: bash smoke_driver.sh <config> <run_dir> main|a2"; exit 2; }
[ -e "$D" ] && { echo "$D exists: pick a fresh run dir"; exit 1; }
python3 $SRC/orchestrator.py init "$D" --config "$CFG" || exit 1
cat > $D/fake_overseer.json <<'EOF'
{"2": {"decision": "NEW_PHASE", "config": {"num_generations": 8, "rollouts_per_update": 32, "hard_share": 0.5, "reset_optimizer": false}},
 "3": {"decision": "REVERT", "target": "e1", "config": {"reset_optimizer": true}}}
EOF
( until grep -q '^STEP 1 ' $D/logs/n0_g3_train_1.log 2>/dev/null; do sleep 2; done
  touch $D/PAUSE $D/pause.mine; echo "[driver $(date +%T)] PAUSE touched after STEP 1" ) & pp=$!
orch "run 1 (pause mid-generation)" fake run "$D" --go
kill $pp 2>/dev/null
{ [ $rc = 3 ] && [ -e $D/pause.mine ]; } || stop "run 1 did not end on the driver's pause: stopping"
rm -f $D/PAUSE $D/pause.mine
orch "run 2 (resume, CONTINUE, NEW_PHASE)" fake run "$D" --go --max-launches 2
[ $rc = 0 ] || stop "run 2 failed: stopping"
orch "run 3 (the new phase's segment)" fake run "$D" --go --halt-before decide
[ $rc = 0 ] || stop "run 3 failed: stopping"
mkdir $D.s6 && cp $D/tree.json $D/root_config.json $D.s6/
orch "S6 (one real overseer call on a copy)" real run "$D.s6" --go --halt-before train
s6=$rc
orch "run 4 (REVERT, last segment)" fake run "$D" --go
[ $rc = 0 ] || stop "run 4 failed: stopping"
echo "S6_EXIT $s6"
stop "main done"
