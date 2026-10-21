#!/bin/bash
# Diagnostic experiments after the 5M FAIL: bounded aim mean (the fix) and a second seed of the 5M config.
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
S=runs/overnight/exp2.status
SRV=../teeworlds-0.7.5/build-release
PY=.venv/bin/python
L=runs/overnight/regress2
mkdir -p $L
log() { echo "$(date '+%F %T') $*" >> $S; }
export TWRL_SERVER_DIR=$SRV
ok=1
for t in ppo env relay; do
  timeout 40m $PY tests/test_$t.py -v > $L/test_$t.log 2>&1; rc=$?; log "test_$t exit $rc"; [ $rc -eq 0 ] || ok=0
done
timeout 20m ../teeworlds-0.7.5/build-test/testrunner > $L/gtest.log 2>&1; rc=$?; log "gtest exit $rc"; [ $rc -eq 0 ] || ok=0
# defaults still reproduce the 5M run's first 30 updates
timeout 15m $PY scripts/train_ppo.py --server-dir $SRV --run-dir runs/overnight/det-check2 --envs 32 --updates 30 --max-minutes 10 \
  --checkpoint-steps 250000 --aim gaussian --seed 1 --threads 6 --compile --opponent initial --aim-target-kl 0.015 > $L/det.log 2>&1
DET=$($PY -c "
import json
a = [json.loads(l) for l in open('runs/overnight/learn-5m-kl015/metrics.jsonl')][:30]
b = [json.loads(l) for l in open('runs/overnight/det-check2/metrics.jsonl')]
skip = {'minutes', 'collect_s', 'train_s'}
print(int(len(b) == 30 and all(x[k] == y.get(k) for x, y in zip(a, b) for k in x if k not in skip)))")
log "determinism vs learn-5m-kl015 first 30 updates: $DET"
[ "$DET" = "1" ] || ok=0
if [ $ok -ne 1 ]; then log "REGRESSION: experiments not started"; exit 1; fi
EVAL=500000,1000000,1500000,2000000,3000000,4000000,5000000
COMMON="--envs 32 --updates 1221 --max-minutes 140 --checkpoint-steps 250000 --aim gaussian --threads 5 --compile --opponent initial --aim-target-kl 0.015"
log "start bounded (seed 1) and seed 2 (raw)"
timeout 200m $PY bench/learn_pipeline.py --run-dir runs/overnight/learn-5m-bounded --eval-at $EVAL -- $COMMON --seed 1 --aim-mean bounded > runs/overnight/learn-5m-bounded.out 2>&1 &
sleep 30
timeout 200m $PY bench/learn_pipeline.py --run-dir runs/overnight/learn-5m-seed2 --eval-at $EVAL -- $COMMON --seed 2 > runs/overnight/learn-5m-seed2.out 2>&1 &
wait
log "both pipelines finished"
ck() { $PY -c "import json,os; print(os.path.join('$1', json.load(open('$1/eval/at_$2.json'))['current'].split('/')[-1]))"; }
for R in runs/overnight/learn-5m-bounded runs/overnight/learn-5m-seed2; do
  CKS="$R/ckpt_0000.pt"
  for t in 500000 1000000 2000000 3000000 4000000 5000000; do [ -f $R/eval/at_$t.json ] && CKS="$CKS $(ck $R $t)"; done
  n=$(basename $R)
  timeout 90m $PY bench/diagnose.py --server-dir $SRV --opponent $R/ckpt_0000.pt --checkpoints $CKS --out runs/overnight/diag_$n.json > runs/overnight/diag_$n.log 2>&1 &
  sleep 20
done
LAST=$(basename $(ck runs/overnight/learn-5m-bounded 5000000))
timeout 60m $PY bench/credit_audit.py --server-dir $SRV --run-dir runs/overnight/learn-5m-bounded --learner $LAST --rollouts 40 --envs 32 --traces 4 --out runs/overnight/audit_bounded_final.json > runs/overnight/audit_bounded_final.log 2>&1 &
wait
log "exp2 done"
