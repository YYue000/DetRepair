#!/usr/bin/env bash
ROOT=../../../../../..

CONFIG=$1
gpuid=$2


export PYTHONPATH=$ROOT:$PYTHONPATH 

#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $ROOT/tools/train.py $CONFIG --work-dir work_dirs --launcher pytorch ${@:3} \
#    2>&1|tee train.log
python $ROOT/tools/train.py \
    ${CONFIG} --work-dir work_dirs --gpu-id $gpuid \
    2>&1|tee -a train.log
