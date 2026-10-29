#!/bin/bash
# ReDo + aim log_std floor: seed 1 resumed from 15M (ckpt_3663) to 30M. The one change against runs/phase3-redo
# (the control) is --aim-log-std-floor -2.3; everything else, watchdog limit 0.25 included, is the same.
# --max-minutes is wall-clock only (the control was stopped by the watchdog, not by time).
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
D=runs/phase3-redo-floor
S=$D/status
SRV=../teeworlds-0.7.5/build-v4
PY=.venv/bin/python
log() { echo "$(date '+%F %T') $*" >> $S; }
COMMON="--envs 32 --updates 7325 --max-minutes 330 --checkpoint-steps 1000000 --aim gaussian --threads 4 --compile --opponent initial --aim-target-kl 0.015 --aim-mean bounded --aim-mask shots --env-timeout 600 --redo-every 25 --redo-tau 0 --aim-log-std-floor -2.3"
log "seed1 start"
timeout 360m $PY bench/learn_pipeline.py --server-dir $SRV --run-dir $D/seed1 --eval-at 20000000,25000000,30000000 --aim-kl-limit 0.25 -- $COMMON --seed 1 --resume $D/seed1/ckpt_3663.pt > $D/seed1.out 2>&1
log "seed1 done exit $?"
R=$D/seed1
CKS=$(ls $R/ckpt_[0-9]*.pt | awk -F'ckpt_|.pt' '$2+0>=4883' | awk 'NR%5==1' | tr '\n' ' ')
$PY bench/collapse_probe.py --server-dir $SRV --run-dir $R --checkpoints $(for c in $CKS; do basename $c; done) --batches 4 --out $D/probe_seed1.json > $D/probe_seed1.log 2>&1
log "probe exit $?"
log "seq done"
