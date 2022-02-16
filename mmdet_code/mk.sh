
for i in $(seq 1 5)
do
    mkdir cfg$i
    cp ../cfg4/cfg5/*.sh cfg$i
    cp ../cfg4/cfg5/get_mean.py cfg$i
    cp ../cfg4/cfg$i/retinanet_r50_fpn_1x_coco.py cfg$i
done

