#!/bin/bash
# 3-seed run, sequential: seed 1, then 2, then 3, each started only after the previous pipeline has
# finished (training, every evaluation and summary.json); then the fire diagnostics. Settings are those
# of run3.sh (aborted-parallel/), only the execution changed: three seeds in parallel overloaded the box.
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
D=runs/phase3-3seed
S=$D/status
SRV=../teeworlds-0.7.5/build-v4
PY=.venv/bin/python
log() { echo "$(date '+%F %T') $*" >> $S; }
EVAL=500000,1000000,1500000,2000000,3000000,4000000,5000000
COMMON="--envs 32 --updates 1221 --max-minutes 1500 --checkpoint-steps 250000 --aim gaussian --threads 4 --compile --opponent initial --aim-target-kl 0.015 --aim-mean bounded --aim-mask shots+hooks --env-timeout 600"
log "seq start (seeds 1 2 3 one after another)"
for seed in 1 2 3; do
  log "seq seed$seed start"
  timeout 1560m $PY bench/learn_pipeline.py --server-dir $SRV --run-dir $D/seed$seed --eval-at $EVAL -- $COMMON --seed $seed > $D/seed$seed.out 2>&1
  log "seq seed$seed done exit $?"
done
ck() { $PY -c "import json,os; print(os.path.join('$1', json.load(open('$1/eval/at_$2.json'))['current'].split('/')[-1]))"; }
for seed in 1 2 3; do
  R=$D/seed$seed
  CKS="$R/ckpt_0000.pt"
  for t in 500000 1000000 2000000 3000000 4000000 5000000; do [ -f $R/eval/at_$t.json ] && CKS="$CKS $(ck $R $t)"; done
  timeout 180m $PY bench/diagnose.py --server-dir $SRV --opponent $R/ckpt_0000.pt --checkpoints $CKS --out $D/diag_seed$seed.json > $D/diag_seed$seed.log 2>&1 &
done
wait
log "seq done"
