from collect.collectFailure import get_model_results as get_failure_results,get_failure_imgid
import imagecorruptions
from mmcv.image import imread, imwrite

import pickle
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s#%(lineno)d:%(message)s')
logger = logging.getLogger('global')

def get_success_imgid(result_file, clean, B=1, key_idx=1):
    results = pickle.load(open(result_file, 'rb'))
    failure_flag = results[key_idx]<clean[key_idx]
    valid_flag = results[0]>0
    flag = (~failure_flag)&valid_flag
    failure_imgid = results[0][flag]
    return success_imgid


def get_distance(imgid, corruption, severity):
    img_path = f'/yueyuxin/data/coco/val2017/{imgid:012}.jpg'
    image = imread(img_path)
    corrupted = imagecorruptions.corrupt(image, corruption_name=corruption, severity=severity)
    return np.mean(np.abs(image-corrupted))

def get_failure_distances(failure_imgid, corruption, severity):
    D = 0
    for imgid in failure_imgid:
        imgid = int(imgid)
        d = get_distance(imgid, corruption, severity)
        D += d
    return D/len(failure_imgid)

if __name__ == '__main__':
    path = '../../retinanet'
    mode = 'failure'
    #mode = 'success'
    if mode == 'failure':
        failure_results = get_failure_results(path, get_failure_func=get_failure_imgid)
    elif mode == 'success':
        failure_results = get_failure_results(path, get_failure_func=get_success_imgid)
    else:
        raise NotImplementedError
    logger.info(f'{path}')
    logger.info('-'*20)

    info = {}

    for m, minfo in failure_results.items():
        for c, cinfo in minfo.items():
            for s, sinfo in cinfo.items():
                d = get_failure_distances(sinfo, c, s)
                logger.info(f'{m},{c},{s},{d},{len(sinfo)}')
                info[f'{m}-{c}-{s}'] = d
    pickle.dump(info, open(f'distances_{mode}_{path.split("/")[-1]}.pkl','wb'))
