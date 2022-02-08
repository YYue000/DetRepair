from pycocotools.coco import COCO

import json
import numpy as np

import itertools

#box_file = '/home/yueyuxin/mmdetection/exp/fpn-r50/baseline/output.bbox.json'
#box_file = '/home/yueyuxin/mmdetection/exp/fpn-r50/fog-5/output.bbox.json'
box_file = '/home/yueyuxin/mmdetection/exp/retina-r50/baseline/output.bbox.json'

from pycocotools.cocoeval import COCOeval
print('import from global')
#from cocoeval import COCOeval
#print('import from local')


anno_file = '/home/yueyuxin/data/coco/annotations/instances_val2017.json'

cocoGt = COCO(anno_file)

metric='bbox'
predictions = json.load(open(box_file))
cocoDt = cocoGt.loadRes(predictions)
cocoEval = COCOeval(cocoGt, cocoDt, iouType=metric)
#cocoEval.params.areaRng = [[0**2, 1e5**2]]
#cocoEval.params.areaRngLbl = ['all']
#cocoEval.params.maxDets = [1, 10]
#cocoEval.params.recThrs = np.linspace(.0, 1.00, 2+1, endpoint=True)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()


imgids = cocoGt.getImgIds().copy()
B=2

ev_res = np.zeros([B+12, int(np.power(len(imgids), B))])
for i,imgid in enumerate(itertools.product(imgids, repeat=B)):
    pred = [_ for _ in predictions if _['image_id'] in imgid]
    if len(pred) == 0: continue
    cocoDt = cocoGt.loadRes(pred)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType=metric)
    #cocoEval.params.maxDets = [1, 10]
    cocoEval.params.imgIds = list(imgid)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    ap0 = cocoEval.stats
    

    """
    #anns = cocoGt.imgToAnns[imgid]
    #N = len([_ for _ in anns if not _['iscrowd']])
    N=10
    cocoEval.params.recThrs = np.linspace(.0, 1.00, N+1, endpoint=True)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    ap = cocoEval.stats
    print('Evaluating',imgid, ap,ap0, ap-ap0)
    """
    print('Evaluating',imgid, ap0)
    ev_res[0:B,i] = imgid
    ev_res[B:B+12,i] = ap0
    #ev_res[1+12:1+12*2,i] = ap0

import pickle
pickle.dump(ev_res, open('eval_res_per_img_batch2_square.pkl','wb'))
