from imagecorruptions import corrupt
import json
import numpy as np
import mmcv
import os

corruption_name = 'fog'
severity = 5

img_root = '/home/yueyuxin/data/coco/val2017/'
imageids = [_['id'] for _ in json.load(open('/home/yueyuxin/data/coco/c_annotations/retina_fog_5_train.json'))['images']]
new_img_root = f'/home/yueyuxin/data/coco/corrupted/{corruption_name}-{severity}/train/'

if not os.path.exists(new_img_root):
    os.makedirs(new_img_root)

for imgid in imageids:
    image = mmcv.imread(img_root+f'{imgid:012}.jpg').astype(np.uint8)
    #image = mmcv.bgr2rgb(image)
    corrupted_image = corrupt(image, corruption_name=corruption_name, severity=severity)
    mmcv.imwrite(corrupted_image, new_img_root+f'{imgid:012}.jpg')
