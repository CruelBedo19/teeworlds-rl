#!/bin/bash
# ReDo test (see bench/results/phase3-long/collapse): seed 1 of the long run resumed from its best checkpoint
# (15M, ckpt_3663) to 30M with one change, --redo-every 25 --redo-tau 0 (recycle units that are zero on the
# whole batch). Everything else, watchdog limit included, as runs/phase3-long attempt 3; that run (no ReDo,
# same start) is the control. Success: trunk2 stays alive at 25-30M and aim/kills hold or beat 15M.
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
D=runs/phase3-redo
S=$D/status
SRV=../teeworlds-0.7.5/build-v4
PY=.venv/bin/python
log() { echo "$(date '+%F %T') $*" >> $S; }
COMMON="--envs 32 --updates 7325 --max-minutes 200 --checkpoint-steps 1000000 --aim gaussian --threads 4 --compile --opponent initial --aim-target-kl 0.015 --aim-mean bounded --aim-mask shots --env-timeout 600 --redo-every 25 --redo-tau 0"
log "seed1 start"
timeout 230m $PY bench/learn_pipeline.py --server-dir $SRV --run-dir $D/seed1 --eval-at 20000000,25000000,30000000 --aim-kl-limit 0.25 -- $COMMON --seed 1 --resume $D/seed1/ckpt_3663.pt > $D/seed1.out 2>&1
log "seed1 done exit $?"
R=$D/seed1
CKS=$(ls $R/ckpt_[0-9]*.pt | awk -F'ckpt_|.pt' '$2+0>=4883' | awk 'NR%5==1' | tr '\n' ' ')
$PY bench/collapse_probe.py --server-dir $SRV --run-dir $R --checkpoints $(for c in $CKS; do basename $c; done) --batches 4 --out $D/probe_seed1.json > $D/probe_seed1.log 2>&1
log "probe exit $?"
log "seq done"
