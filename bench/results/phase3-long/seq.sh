#!/bin/bash
# Overnight long run, see PLAN.md. Shots-only baseline resumed 6M -> 40M; seed 3 then seed 1.
# Attempt 3: attempt 2 (attempt2/) was stopped again by aim-KL spikes (single updates 0.9-1.3 as the aim std
# narrows) while aim cos and hit rate kept rising; each seed resumes from its last checkpoint, limit 0.25.
# Attempt 2: attempt 1 (attempt1/) was stopped at 7M by a single aim-KL spike tripping the 0.05 watchdog;
# both seeds restart from their 6M checkpoint with the watchdog's aim-KL limit at 0.10. Deadline 11:13 (12 h).
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
D=runs/phase3-long
S=$D/status
SRV=../teeworlds-0.7.5/build-v4
PY=.venv/bin/python
T0=$(date -d '2026-10-26 23:13' +%s)
log() { echo "$(date '+%F %T') $*" >> $S; }
EVAL=10000000,15000000,20000000,25000000,30000000,35000000,40000000
COMMON="--envs 32 --updates 9766 --max-minutes 330 --checkpoint-steps 1000000 --aim gaussian --threads 4 --compile --opponent initial --aim-target-kl 0.015 --aim-mean bounded --aim-mask shots --env-timeout 600"
log "seq start, attempt 3 (seed 3 from 8M, seed 1 from 15M, to 40M, deadline 11:13)"
for seed in 3 1; do
  left=$(( 43200 - ($(date +%s) - T0) ))
  if [ $left -lt 10800 ]; then log "seq seed$seed skipped: only $((left/60)) min left"; continue; fi
  log "seq seed$seed start"
  timeout 340m $PY bench/learn_pipeline.py --server-dir $SRV --run-dir $D/seed$seed --eval-at $EVAL --aim-kl-limit 0.25 -- $COMMON --seed $seed --resume $(ls $D/seed$seed/ckpt_[0-9]*.pt | tail -n1) > $D/seed$seed.out 2>&1
  log "seq seed$seed done exit $?"
done
for seed in 3 1; do
  R=$D/seed$seed
  CKS=""
  for u in 2442 4883 7325 9766; do f=$(ls $R/ckpt_*.pt 2>/dev/null | grep -v final | awk -F'ckpt_|.pt' -v u=$u '$2+0>=u{print; exit}'); [ -n "$f" ] && CKS="$CKS $f"; done
  [ -z "$CKS" ] && continue
  left=$(( 43200 - ($(date +%s) - T0) ))
  if [ $left -lt 1800 ]; then log "diagnose seed$seed skipped: $((left/60)) min left"; continue; fi
  timeout $((left/60 - 5))m $PY bench/diagnose.py --server-dir $SRV --opponent $R/ckpt_0000.pt --checkpoints $(echo $CKS) --out $D/diag_seed$seed.json > $D/diag_seed$seed.log 2>&1
  log "diagnose seed$seed exit $?"
done
log "seq done"
