import mmcv
import numpy as np
def cmp(d1, d2):
    #print(type(d1), type(d2))
    if type(d1)!=type(d2):
        return False
    if d1 is None:
        return True
    elif isinstance(d1, list) or isinstance(d1, tuple):
        if len(d1)!=len(d2):
            return False
        for _,__ in zip(d1,d2):
            if not cmp(_, __):
                return False
        return True
    elif isinstance(d1, str): 
        return d1==d2
    elif not isinstance(d1, dict):
        return np.sum(np.abs(d1-d2))<1e-5
    else:
        f = True
        itd = (set(d1.keys())|set(d2.keys()))-(set(d1.keys())&set(d2.keys()))
        if len(itd) > 0:
            print(set(d1.keys()-set(d2.keys())), set(d2.keys())-set(d1.keys()))
        for k in set(d1.keys())&set(d2.keys()):
            c = cmp(d1[k], d2[k])
            if not c:
                print('st'+'-'*20)
                print(k)
                print(d1[k])
                print(d2[k])
                print('end'+'-'*20)
                f = False
        return f

if __name__ == '__main__':
    from mmcv import Config
    cfg0 = Config.fromfile('../retinanet_r50_fpn_1x_coco.py')
    #cfg0 = dict(Config.fromfile('../retinanet_r50_fpn_1x_coco.py'))    
    cfg6 = Config.fromfile('retinanet_r50_fpn_1x_coco.py')
    cfg0 = dict(cfg0)
    cfg6 = dict(cfg6)
    print(cmp(cfg0,cfg6))
            
