import pickle
import os
import numpy as np
import json
from collect.collectFailure import get_model_results
from table.utils import output_table
from functools import partial


areaRngLbl = ['all', 'small', 'medium', 'large']
areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]

def get_failure_imgid(result_file, clean, B=1, key_idx=1):
    results = pickle.load(open(result_file, 'rb'))
    failure_flag = results[key_idx]<clean[key_idx]
    failure_imgid = results[0][failure_flag]
    return failure_imgid

def get_stats_failure_dataset(anns, failure_imgid):
    fanns = [_ for _ in anns['annotations'] if _['image_id'] in failure_imgid]
    
    acnt = {_:0 for _ in areaRngLbl}
    crowd = 0
    inst_num = 0

    category_num = {}
    for ann in fanns:
        if 'ignore' in ann and ann['ignore']:
            continue
        inst_num += 1

        if ann['iscrowd']:
            crowd += 1
            continue

        for i,area in enumerate(areaRng):
            if ann['area']>area[0] and ann['area']<area[1]:
                acnt[areaRngLbl[i]] += 1
        
        if ann['category_id'] not in category_num:
            category_num[ann['category_id']] = 0

        category_num[ann['category_id']] += 1
    return {'instance_num':inst_num, 'crowd_num':crowd, 'category_num':category_num, 'area_num':acnt}

def get_failure_stats(path, clean, anns):
    imgid = get_failure_imgid(path, clean)
    return get_stats_failure_dataset(anns, imgid)

def _ap_str_crowd(x):
    r = x['crowd_num']/x['instance_num']*100
    return f"&{r:.2f}"

def _ap_str_area(lbl,x):
    r = x['area_num'][lbl]/x['instance_num']*100
    return f"&{r:.2f}"


if __name__ == '__main__':
    ann_file = '/yueyuxin/data/coco/annotations/instances_val2017.json'
    anns = json.load(open(ann_file))

    results = get_model_results('../../retinanet', get_failure_func=partial(get_failure_stats, anns=anns))

    for lbl in areaRngLbl:
        tb = output_table(results, f'area_{lbl}', lambda x:_ap_str_area(lbl,x))
        with open(f'area_{lbl}.txt','w') as fw:
            fw.write(tb)
