from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import numpy as np
box_file = '/home/yueyuxin/mmdetection/exp/retina-r50/baseline/output.bbox.json'
anno_file = '/home/yueyuxin/data/coco/annotations/instances_val2017.json'


cocoGt = COCO(anno_file)

metric='bbox'
predictions = json.load(open(box_file))
cocoDt = cocoGt.loadRes(predictions)
cocoEval = COCOeval(cocoGt, cocoDt, iouType=metric)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()


imgids = cocoGt.getImgIds().copy()

ev_res = np.zeros([12,len(imgids)])
for i,imgid in enumerate(imgids):
    pred = [_ for _ in predictions if _['image_id']==imgid]
    cocoDt = cocoGt.loadRes(pred)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType=metric)
    cocoEval.params.imgIds = [imgid]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    ap = cocoEval.stats
    print('Evaluating',imgid, ap)
    ev_res[:,i] = ap

import pickle
pickle.dump(ev_res, open('eval_res_per_img.pkl','wb'))
