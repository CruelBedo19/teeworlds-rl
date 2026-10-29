#!/bin/bash
# ReDo + aim log_std floor, rerun with the exact-KL watchdog (commit 1bb9a25): seed 1 resumed from 15M (ckpt_3663) to 30M.
# Training settings are those of runs/phase3-redo-floor; the only change is that the watchdog's 20-update aim KL mean
# (limit 0.25, unchanged) reads aim_kl_exact instead of the minibatch estimate aim_approx_kl.
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
D=runs/phase3-redo-floor-exactkl
S=$D/status
SRV=../teeworlds-0.7.5/build-v4
PY=.venv/bin/python
log() { echo "$(date '+%F %T') $*" >> $S; }
COMMON="--envs 32 --updates 7325 --max-minutes 330 --checkpoint-steps 1000000 --aim gaussian --threads 4 --compile --opponent initial --aim-target-kl 0.015 --aim-mean bounded --aim-mask shots --env-timeout 600 --redo-every 25 --redo-tau 0 --aim-log-std-floor -2.3"
log "seed1 start $(git rev-parse --short HEAD)"
timeout 360m $PY bench/learn_pipeline.py --server-dir $SRV --run-dir $D/seed1 --eval-at 20000000,25000000,30000000 --aim-kl-limit 0.25 -- $COMMON --seed 1 --resume $D/seed1/ckpt_3663.pt > $D/seed1.out 2>&1
log "seed1 done exit $?"
R=$D/seed1
# 20M, 25M, 30M (every 5th checkpoint from 4883) and the last one, if the run stopped early
CKS=$( (ls $R/ckpt_[0-9]*.pt | awk -F'ckpt_|.pt' '$2+0>=4883' | awk 'NR%5==1'; ls $R/ckpt_[0-9]*.pt | tail -1) | sort -u | xargs -n1 basename)
if [ -n "$CKS" ]; then
  $PY bench/collapse_probe.py --server-dir $SRV --run-dir $R --checkpoints $CKS --batches 4 --out $D/probe_seed1.json > $D/probe_seed1.log 2>&1
  log "probe exit $?"
else
  log "probe skipped: no checkpoint past 20M"
fi
log "seq done"
