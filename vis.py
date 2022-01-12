from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import numpy as np
import json

config_file = '../retinanet_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '../retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
device = 'cuda:1'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)


anno_file = '/home/yueyuxin/data/coco/annotations/instances_val2017.json'
anns = json.load(open(anno_file))
catids = sorted(list(set([_['category_id'] for _ in anns['annotations']])))  
catid2idx = {_:i for i,_ in enumerate(catids)}

# inference the demo image
#imgid = 458054
imgid_list = [506656, 143572, 103548, 70774, 2153, 477441, 296657, 527029]

def xywh2xyxy(bbox):
    if bbox.shape[0]== 0:return bbox
    nbbox = bbox.copy()
    nbbox[:,2] = bbox[:,2]+bbox[:,0]
    nbbox[:,3] = bbox[:,3]+bbox[:,1]
    return nbbox

def draw(imgid):
    img_path = f'/home/yueyuxin/data/coco/val2017/{imgid:012}.jpg'

    result = inference_detector(model, img_path)
    model.show_result(img_path, result, show=True, out_file=f'{imgid:012}_retina.jpg')

    gt = [[] for _ in range(80)]

    cnt = 0
    for _ in anns['annotations']:
        if _['iscrowd'] == 1:
            continue
        if _['image_id']==imgid:
            gt[catid2idx[_['category_id']]].append(_['bbox']+[1.0])
            cnt += 1
    print(imgid,cnt, sum([_.shape[0] for _ in result]))
    gt=[xywh2xyxy(np.array(_).reshape(-1,5)) for _ in gt]
    model.show_result(img_path, gt, show=True, out_file=f'{imgid:012}_gt.jpg')

for imgid in imgid_list:
    draw(imgid)

