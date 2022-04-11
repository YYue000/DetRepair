#!/usr/bin/env bash
ROOT=../../../../../..

CONFIG=$1
GPUS=4
PORT=${PORT:-29500}


export PYTHONPATH=$ROOT:$PYTHONPATH 

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $ROOT/tools/train.py $CONFIG --work-dir work_dir --launcher pytorch ${@:3} \
    2>&1|tee train.log
#gpuid=$3
#python $ROOT/tools/train.py \
#    ${CONFIG} --work-dir work-dir --gpu-id $gpuid \
#    2>&1|tee train.log
