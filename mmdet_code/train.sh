#!/usr/bin/env bash
ROOT=$1

CONFIG=$2

gpuid=$3

export PYTHONPATH=$ROOT:$PYTHONPATH 
EVAL_METRICS=bbox

python $ROOT/tools/train.py \
    ${CONFIG} --gpu-id $gpuid \
    2>&1|tee train.log
