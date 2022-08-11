#!/usr/bin/env bash
ROOT=../../../../../..

CONFIG=$1
MAXEPOCH=$2
gpuid=$3
MINEPOCH=${4-1}
echo $CONFIG

export PYTHONPATH=$ROOT:$PYTHONPATH 
EVAL_METRICS=bbox

for e in $(seq $MINEPOCH $MAXEPOCH)
do
    CHECKPOINT=work_dirs/epoch_${e}.pth
    python $ROOT/tools/test.py \
        ${CONFIG} ${CHECKPOINT} --out output/output.pkl --eval $EVAL_METRICS --tmpdir output --gpu-id $gpuid \
        2>&1|tee -a test.log
done
