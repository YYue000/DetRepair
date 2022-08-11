g=$1
c='contrast'
resume_iter=1260
start_iter=${resume_iter}
python ../cal.py --model resnet101 --corruption $c --prob 0.5 --batch_size 32 --gpu $g --ckpt_save_path checkpoints_${c} --train_print_freq 10 --ckpt_freq 20 \
    --load_from checkpoints_${c}/ckpt_iter${resume_iter}.pth --start_iter ${start_iter} --max_iter 2000 2>&1|tee -a bn_${c}.log
