#!/bin/bash
# Regression suite, then (only if everything passes) the 5M frozen-opponent run with aim_target_kl 0.015.
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
L=runs/overnight/regress
mkdir -p $L
export TWRL_SERVER_DIR=../teeworlds-0.7.5/build-release
ok=1
for t in ppo env relay; do
  timeout 40m .venv/bin/python tests/test_$t.py -v > $L/test_$t.log 2>&1; rc=$?
  echo "test_$t exit $rc" >> $L/status; [ $rc -eq 0 ] || ok=0
done
timeout 20m ../teeworlds-0.7.5/build-test/testrunner > $L/gtest.log 2>&1; rc=$?
echo "gtest exit $rc" >> $L/status; [ $rc -eq 0 ] || ok=0
if [ $ok -ne 1 ]; then echo "REGRESSION: 5M not started" >> $L/status; exit 1; fi
echo "all passed $(date)" >> $L/status
timeout 120m .venv/bin/python bench/learn_pipeline.py --run-dir runs/overnight/learn-5m-kl015 \
  --eval-at 500000,1000000,1500000,2000000,3000000,4000000,5000000 -- \
  --envs 32 --updates 1221 --max-minutes 100 --checkpoint-steps 250000 --aim gaussian --seed 1 --threads 6 --compile \
  --opponent initial --aim-target-kl 0.015 > runs/overnight/learn-5m-kl015.out 2>&1
echo "pipeline exit $? $(date)" >> $L/status
