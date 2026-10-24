#!/bin/bash
# Mask A/B, arm shots-only: the aim learns only from shot decisions; everything else as
# phase3-3seed/seq.sh (shots+hooks), which is the control. Seeds 1, 2, 3 one after another, then the
# fire diagnostics.
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
D=runs/phase3-ab-mask
S=$D/status
SRV=../teeworlds-0.7.5/build-v4
PY=.venv/bin/python
log() { echo "$(date '+%F %T') $*" >> $S; }
EVAL=500000,1000000,1500000,2000000,3000000,4000000,5000000
COMMON="--envs 32 --updates 1221 --max-minutes 1500 --checkpoint-steps 250000 --aim gaussian --threads 4 --compile --opponent initial --aim-target-kl 0.015 --aim-mean bounded --aim-mask shots --env-timeout 600"
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
