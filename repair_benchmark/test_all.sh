#!/usr/bin/env bash
ROOT=../../../../..

CONFIG=$1
MAXEPOCH=$2
echo $CONFIG

export PYTHONPATH=$ROOT:$PYTHONPATH 
EVAL_METRICS=bbox

for e in $(seq 1 $MAXEPOCH)
do
    CHECKPOINT=work_dirs/epoch_${e}.pth
    python $ROOT/tools/test.py \
        ${CONFIG} ${CHECKPOINT} --out output/output.pkl --eval $EVAL_METRICS --tmpdir output \
        2>&1|tee -a test.log
done
