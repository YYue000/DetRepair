#!/usr/bin/env bash
ROOT=../../..

CONFIG=$1
CHECKPOINT=$2
MODE=$3
echo $CONFIG

export PYTHONPATH=$ROOT:$PYTHONPATH 
EVAL_METRICS=bbox

python $ROOT/tools/test.py \
    ${CONFIG} ${CHECKPOINT} --out output/output.pkl --eval $EVAL_METRICS --tmpdir output \
    2>&1|tee test_${MODE}.log
