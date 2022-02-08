import json
import numpy as np
import os
import copy

corruption_name = 'fog'
severity = 5
root_corrupted='corrupted/fog-5/train/'
root_clean = 'val2017/'
corrupted = json.load(open('/home/yueyuxin/data/coco/c_annotations/retina_fog_5_train.json'))

clean_all = json.load(open('/home/yueyuxin/data/coco/annotations/instances_val2017.json'))

# reference clean
imgids = [_['id'] for _ in corrupted['images']]
clean = {'annotations':[_ for _ in clean_all['annotations'] if _['image_id'] in imgids]}

merge = {k:v for k,v in corrupted.items() if k not in ['images','annotations']}
merge['images'] = []
merge['annotations'] = clean['annotations']

def get_imgid2anns(anns):
    g = {}
    for ann in anns:
        if ann['image_id'] not in g:
            g[ann['image_id']] = []
        g[ann['image_id']].append(ann)
    return g

corrupted_imgid2anns = get_imgid2anns(corrupted['annotations'])

for img in corrupted['images']:
    cln_img = copy.deepcopy(img)
    cln_img['file_name'] = root_clean+img['file_name']
    merge['images'].append(cln_img)

    imgid = img['id']
    img['file_name'] = root_corrupted+img['file_name']
    img['id'] = -imgid
    #img['id'] = int(imgid+1e6)
    merge['images'].append(img)

    for ann in corrupted_imgid2anns[imgid]:
        ann['image_id'] =  -imgid
        #ann['image_id'] =  int(imgid+1e6)
        ann['id'] = -ann['id']
    merge['annotations'] += corrupted_imgid2anns[imgid]

print(len(merge['images']), len(corrupted['images']))

json.dump(merge, open('retina_fog_5_1k_clean_ref_1k.json','w'))


