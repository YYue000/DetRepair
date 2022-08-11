g=$1
for c in fog
do
    python ../cal.py --model resnet101 --corruption $c --prob 0.5 --batch_size 32 --gpu $g --ckpt_save_path checkpoints_${c} --train_print_freq 10 --ckpt_freq 20 --load_from ../resnet101_clear_bn.pth --max_iter 2000 2>&1|tee bn_${c}.log
done
