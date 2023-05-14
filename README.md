# DetRepair

These are the scripts used in *How to Repair Object Detectors? A Benchmark, an Empirical Study, and a Way Forward*.

<img width="1467" alt="image" src="https://github.com/YYue000/DetRepair/assets/25451529/d3f02c06-e0b0-43ad-b9f0-459383efbea2">


## Dependencies

https://github.com/YYue000/mmdetection/tree/repair

https://github.com/YYue000/Neighbor2Neighbor

## Failure

Corruptions are available in imagecorruptions[https://github.com/bethgelab/imagecorruptions].
Models are based on mmdetection[https://github.com/YYue000/mmdetection/tree/repair].

Failure samples are collected with scripts in benchmark/.
We first compute image-wise AP with benchmark/setup/eval_one_img.py.
Then we setup the failure dataset with benchmark/setup/setup.py

Analysis for failure samples are conducted with scripts in benchmark/post.

## Repair Methods
To automatically setup the benchmark with many experiments, please refer to repair_benchmark/post/setup.py.

### Fine-tuning & Augmentations

Directly applied with mmdetection. Scripts are availble in repair_benchmark/\*.sh.

### Denoising

We use a refactored version of Neighbor2Neighbor[https://github.com/YYue000/Neighbor2Neighbor].

### BN Calibration

Please refer to imagenet_bn_calibration/cal.py.
