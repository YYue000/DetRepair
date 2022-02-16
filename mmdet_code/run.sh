ROOT=../../../..
sh train.sh $ROOT retinanet_r50_fpn_1x_coco.py $1

sh test_all.sh $ROOT retinanet_r50_fpn_1x_coco.py  8 fog5 $1

python get_mean.py 2>&1|tee sum.log
