from pycocotools.coco import COCO

import json
import pickle

import os
import sys

import itertools
import numpy as np


#box_file = '/home/yueyuxin/mmdetection/exp/fpn-r50/baseline/output.bbox.json'
#box_file = '/home/yueyuxin/mmdetection/exp/fpn-r50/fog-5/output.bbox.json'
#box_file = '/home/yueyuxin/mmdetection/exp/retina-r50/baseline/output.bbox.json'


from pycocotools.cocoeval import COCOeval
print('import from global')
#from cocoeval import COCOeval
#print('import from local')


anno_file = '/yueyuxin/data/coco/annotations/instances_val2017.json'

def get_per_img_ap(box_file, B=1):
    result_file = box_file.replace('.bbox.json', '.ap.pkl')
    if os.path.exists(result_file): return 
    print(box_file, result_file)

    cocoGt = COCO(anno_file)

    metric='bbox'
    predictions = json.load(open(box_file))
    cocoDt = cocoGt.loadRes(predictions)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType=metric)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


    imgids = cocoGt.getImgIds().copy()

    ev_res = np.zeros([B+12, int(np.power(len(imgids), B))])
    for i,imgid in enumerate(itertools.product(imgids, repeat=B)):
        pred = [_ for _ in predictions if _['image_id'] in imgid]
        if len(pred) == 0: continue
        cocoDt = cocoGt.loadRes(pred)
        cocoEval = COCOeval(cocoGt, cocoDt, iouType=metric)
        cocoEval.params.imgIds = list(imgid)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        ap0 = cocoEval.stats
        
        print('Evaluating',imgid, ap0)
        ev_res[0:B,i] = imgid
        ev_res[B:B+12,i] = ap0

    pickle.dump(ev_res, open(result_file,'wb'))

if __name__=='__main__':

    box_file = sys.argv[1]
    if os.path.isfile(box_file):
        get_per_img_ap(box_file)
    else:
        for root, d, files in os.walk(box_file):
            for f in files:
                p = os.path.join(root, f)
                print(p)
                if p.endswith('.bbox.json'):
                    get_per_img_ap(p)

