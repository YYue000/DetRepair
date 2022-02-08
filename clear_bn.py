import torch
import os

path = '/home/yueyuxin/mmdetection/exp/retina-r50/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
model = torch.load(path, map_location=torch.device('cpu'))

stats_ks = {'running_mean': 0, 'running_var': 1, 'num_batches_tracked':0}
params_ks = {'weight': 1, 'bias': 0}

def search_d(d,k,v):
    for _,__ in d.items():
        if _ in k:
            if __ == 0:
                return torch.zeros(v.shape)
            else:
                return torch.ones(v.shape)*__
    return None

for k,v in model['state_dict'].items():
    if 'bn' not in k:
        continue

    nv = search_d(stats_ks, k, v)
    if nv is not None:
        model['state_dict'][k] = nv
        continue

    """
    nv = search_d(params_ks, k, v)
    if nv is not None:
        model['state_dict'][k] = nv
        continue
    """

torch.save(model,path.replace('.pth', '_clear_bn_stats.pth'))
#torch.save(model,path.replace('.pth', '_clear_bn.pth'))
