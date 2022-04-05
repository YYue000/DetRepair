import yaml
import os

def get_models(metafile):
    f = yaml.safe_load(open(metafile))
    det_models = []
    for m in f['Models']:
        if sum([_ in m["Name"] for _ in ['retinanet', 'faster_rcnn', 'cascade_rcnn', 'mask_rcnn']])>0  and 'caffe' in m['Name']: continue
        flag = False
        for r in m['Results']:
            if r['Task'] == 'Object Detection':
                flag = True
                break
        if flag:
            det_models.append(m)
            
    return det_models

if __name__ == '__main__':
    # all in list: ['faster_rcnn', 'retinanet', 'mask_rcnn']
    root = '/home/yueyuxin/mmdetection/configs'
    modelzoo = []
    dt = 0
    for d in os.listdir(root):
        if d in ['panoptic_fpn', 'gn', 'gn+ws', 'seesaw_loss', 'dcn']: continue
        p = os.path.join(root, d, 'metafile.yml')
        if not os.path.exists(p): continue
        models = get_models(p)
        if len(models)==0: continue
        print(d, len(models))
        dt += 1
        modelzoo += models
    print(len(modelzoo), [_['Name'] for _ in modelzoo])
    print(dt)
