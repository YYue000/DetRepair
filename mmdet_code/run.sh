# sh train.sh retinanet_r50_fpn_1x_coco.py

sh test_all.sh retinanet_r50_fpn_1x_coco.py 8 fog5

python get_mean.py 2>&1|tee sum.log
