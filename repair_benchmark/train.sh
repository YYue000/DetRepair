#!/usr/bin/env bash
ROOT=../../../../..

CONFIG=$1
gpuid=$2

export PYTHONPATH=$ROOT:$PYTHONPATH 
EVAL_METRICS=bbox

python $ROOT/tools/train.py \
    ${CONFIG} --work-dir work-dir --gpu-ids $gpuid \
    2>&1|tee train.log
