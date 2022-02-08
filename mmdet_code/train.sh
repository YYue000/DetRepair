#!/usr/bin/env bash
ROOT=../../..

CONFIG=$1


export PYTHONPATH=$ROOT:$PYTHONPATH 
EVAL_METRICS=bbox

python $ROOT/tools/train.py \
    ${CONFIG} --gpu-ids 1 \
    2>&1|tee train.log
