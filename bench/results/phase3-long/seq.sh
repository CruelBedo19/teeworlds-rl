#!/bin/bash
# Overnight long run, see PLAN.md. Shots-only baseline resumed 5M -> 40M; seed 3 then seed 1.
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
D=runs/phase3-long
S=$D/status
SRV=../teeworlds-0.7.5/build-v4
PY=.venv/bin/python
T0=$(date +%s)
log() { echo "$(date '+%F %T') $*" >> $S; }
EVAL=5000000,10000000,15000000,20000000,25000000,30000000,35000000,40000000
COMMON="--envs 32 --updates 9766 --max-minutes 330 --checkpoint-steps 1000000 --aim gaussian --threads 4 --compile --opponent initial --aim-target-kl 0.015 --aim-mean bounded --aim-mask shots --env-timeout 600"
log "seq start (seed 3 then seed 1, 5M -> 40M, cap 12 h)"
for seed in 3 1; do
  left=$(( 43200 - ($(date +%s) - T0) ))
  if [ $left -lt 19800 ]; then log "seq seed$seed skipped: only $((left/60)) min left"; continue; fi
  log "seq seed$seed start"
  timeout 340m $PY bench/learn_pipeline.py --server-dir $SRV --run-dir $D/seed$seed --eval-at $EVAL -- $COMMON --seed $seed --resume $D/seed$seed/ckpt_1221.pt > $D/seed$seed.out 2>&1
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
