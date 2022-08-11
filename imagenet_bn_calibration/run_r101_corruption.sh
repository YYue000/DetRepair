g=$1
for c in shot_noise
do
    python ../cal.py --model resnet101 --corruption $c --batch_size 32 --gpu $g --ckpt_save_path checkpoints_${c} --train_print_freq 10 --ckpt_freq 20 --load_from ../resnet101_clear_bn.pth --max_iter 2000 --aug_data_root ../corruptions/data_${c} 2>&1|tee bn_${c}.log
done
