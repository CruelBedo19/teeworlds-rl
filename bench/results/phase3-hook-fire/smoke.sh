#!/bin/bash
# Short smoke runs after the hook-gradient fix: same seed and settings, aim mask "shots" vs "shots+hooks".
# 250 updates (about 1M learner steps) each, in parallel; then a credit audit of each final checkpoint.
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
SRV=../teeworlds-0.7.5/build-v4
PY=.venv/bin/python
L=runs/phase3-fix
log() { echo "$(date '+%F %T') $*" >> $L/smoke.status; }
COMMON="--server-dir $SRV --envs 32 --updates 250 --max-minutes 40 --checkpoint-steps 250000 --aim gaussian --seed 1 --threads 5 --compile --opponent initial --aim-target-kl 0.015 --aim-mean bounded"
log "start"
timeout 60m $PY scripts/train_ppo.py $COMMON --run-dir $L/smoke-shots --aim-mask shots > $L/smoke-shots.log 2>&1 &
timeout 60m $PY scripts/train_ppo.py $COMMON --run-dir $L/smoke-hooks --aim-mask shots+hooks > $L/smoke-hooks.log 2>&1 &
wait
log "training done"
for r in shots hooks; do
  last=$(ls $L/smoke-$r/ckpt_*.pt | sort | tail -1)
  timeout 40m $PY bench/credit_audit.py --server-dir $SRV --run-dir $L/smoke-$r --learner $(basename $last) --rollouts 20 --envs 32 --traces 2 --out $L/audit-$r.json > $L/audit-$r.log 2>&1 &
done
wait
log "smoke done"
