#!/usr/bin/env bash
ROOT=$1

CONFIG=$2
MAXEPOCH=$3
MODE=$4
gpuid=$5
echo $CONFIG

export PYTHONPATH=$ROOT:$PYTHONPATH 
EVAL_METRICS=bbox

for e in $(seq 1 $MAXEPOCH)
do
    CHECKPOINT=work_dirs/retinanet_r50_fpn_1x_coco/epoch_${e}.pth
    python $ROOT/tools/test.py \
        ${CONFIG} ${CHECKPOINT} --out output/output.pkl --eval $EVAL_METRICS --tmpdir output \
        --gpu-id $gpuid \
        2>&1|tee -a test_${MODE}.log
done
