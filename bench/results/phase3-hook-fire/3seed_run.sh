#!/bin/bash
# 3-seed controlled experiment: bounded aim mean + hook launches in the aim loss, protocol 4 server,
# each seed against its own frozen ckpt_0000, 5M learner steps, same evaluation protocol and criteria.
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
D=runs/phase3-3seed
S=$D/status
SRV=../teeworlds-0.7.5/build-v4
PY=.venv/bin/python
log() { echo "$(date '+%F %T') $*" >> $S; }
EVAL=500000,1000000,1500000,2000000,3000000,4000000,5000000
COMMON="--envs 32 --updates 1221 --max-minutes 240 --checkpoint-steps 250000 --aim gaussian --threads 4 --compile --opponent initial --aim-target-kl 0.015 --aim-mean bounded --aim-mask shots+hooks"
log "start seeds 1 2 3"
for seed in 1 2 3; do
  timeout 330m $PY bench/learn_pipeline.py --server-dir $SRV --run-dir $D/seed$seed --eval-at $EVAL -- $COMMON --seed $seed > $D/seed$seed.out 2>&1 &
  sleep 30
done
wait
log "pipelines finished"
ck() { $PY -c "import json,os; print(os.path.join('$1', json.load(open('$1/eval/at_$2.json'))['current'].split('/')[-1]))"; }
for seed in 1 2 3; do
  R=$D/seed$seed
  CKS="$R/ckpt_0000.pt"
  for t in 500000 1000000 2000000 3000000 4000000 5000000; do [ -f $R/eval/at_$t.json ] && CKS="$CKS $(ck $R $t)"; done
  timeout 120m $PY bench/diagnose.py --server-dir $SRV --opponent $R/ckpt_0000.pt --checkpoints $CKS --out $D/diag_seed$seed.json > $D/diag_seed$seed.log 2>&1 &
done
wait
log "done"
