import json
import yaml
import pickle
import shutil
import os
import copy

import numpy as np

from imagecorruptions import get_corruption_names
from mmcv import Config


CFG_ROOT = '/home/yueyuxin/mmdetection'
DATA_ROOT = '/yueyuxin/data/coco'
WORKSPACEROOT = '/yueyuxin/mmdetection/corruption_benchmarks/'

SET2RUN = lambda x:x
 
TEST_BASE_ROOT='/yueyuxin/mmdetection/corruption_benchmarks'
TEST_IMG_AP_C_FILE_NAME = 'output.pkl.ap.pkl'
TEST_IMG_AP_CLEAN_FILE_NAME = 'output.ap.pkl'
FAILURE_ANN_NAME = 'failure_annotation.json'
FL_TRAIN_NAME = FAILURE_ANN_NAME.replace('.json','_fltrain.json')
FL_TEST_NAME = FAILURE_ANN_NAME.replace('.json','_fltest.json')
#SAMPLE_CLEAN_TRAINING_PATH = '/yueyuxin/data/coco/sampled_annotations/instances_train2017_sample1000.json'
#FL_MG_TRAIN_NAME = FAILURE_ANN_NAME.replace('.json','_fltrain_cltrain.json')

def get_coco_annotations(path):
    return json.load(open(path))
coco_train_annotations = get_coco_annotations('/yueyuxin/data/coco/annotations/instances_train2017.json')
coco_val_annotations = get_coco_annotations('/yueyuxin/data/coco/annotations/instances_val2017.json')

################################################
################################################


def get_model_files(meta_file):
    res = yaml.safe_load(open(meta_file))
    return res['Models']


def get_clean(test_base_dir):
    clean_path = os.path.join(test_base_dir, 'clean', TEST_IMG_AP_CLEAN_FILE_NAME)
    if not os.path.exists(clean_path):
        print(f'Error: {clean_path} not exist')
        return None
    return pickle.load(open(clean_path, 'rb'))

def get_failure_imgid(result_file, clean, B=1, key_idx=1):
    results = pickle.load(open(result_file, 'rb'))
    failure_flag = results[key_idx]<clean[key_idx]
    failure_imgid = results[0][failure_flag]
    return failure_imgid

def parse_failure(path, clean):
    if not os.path.exists(path):
        print(f'Error: file not exist {path}')
        return None
    failure_imgid = get_failure_imgid(path, clean)
    return failure_imgid

def get_failure_set(raw_annotations, failure_imgid, name, image_root=None, dump=True):
    annotations = {_:raw_annotations[_] for _ in ['licenses','info', 'categories']}
    failure_imgid = list(failure_imgid)
    images = []
    for _ in raw_annotations['images']:
        if _['id'] in failure_imgid:
            if image_root is not None:
                img = copy.deepcopy(_)
                img['file_name'] = os.path.join(image_root, img['file_name'])
            else:
                img = _
            images.append(img)

    annotations['images'] = images
    annotations['annotations'] = [_ for _ in raw_annotations['annotations'] if _['image_id'] in failure_imgid]
    if dump:
        json.dump(annotations, open(name, 'w'))
    return annotations

def split_train_test(failure_imgid, trainsize=1000):
    np.random.shuffle(failure_imgid)
    train = failure_imgid[:trainsize]
    test = failure_imgid[trainsize:]
    return train,test

def sample_train(raw_annotations, samplesize=1000):
    annotations = {}
    sample = np.random.choice(np.arange(len(annotations['images'])), samplesize)
    annotations['images'] = [_ for _ in raw_annotations['images'] if _['id'] in sample]
    annotations['annotations'] = [_ for _ in raw_annotations['annotations'] if _['image_id'] in sample]
    return annotations

def merge_samples(base_annotations, app_ann_list, name, dump=True):
    for ann in app_ann_list:
        for k in ['images', 'annotations']:
            base_annotations[k] += app[k]
    if dump:
        json.dump(base_annotations, open(name, 'w'))

def setup_failure_set(failure_imgid, wksp, sample_clean=False):
    name = os.path.join(wksp, FAILURE_ANN_NAME)
    get_failure_set(coco_val_annotations, failure_imgid, name)
    train, test = split_train_test(failure_imgid)

    train_name = os.path.join(wksp, FL_TRAIN_NAME)
    train_ann = get_failure_set(coco_val_annotations, train, train_name)

    test_name = os.path.join(wksp, FL_TEST_NAME)
    get_failure_set(coco_val_annotations, test, test_name)

    if sample_clean:
        #clean = sample_train(coco_train_annotations)
        clean = get_coco_annotations(SAMPLE_CLEAN_TRAINING_PATH)
        mg_name = os.path.join(wksp, FL_MG_TRAIN_NAME)
        merge_samples(train_ann, [clean], mg_name)


def setup_cfg(model, corruption, severity, new_cfg_file, twksp, weight_path):
    cfg_path = os.path.join(CFG_ROOT, model["Config"])  
    cfg = Config.fromfile(cfg_path)
    twksp = SET2RUN(twksp)

    aug = dict(type='Corrupt',corruption=corruption, severity=severity)


    assert cfg['data']['val']['type'] == 'CocoDataset'
    assert cfg['data']['test']['type'] == 'CocoDataset'
    cfg['data']['test']['ann_file'] = os.path.join(twksp, FL_TEST_NAME)
    cfg['data']['test']['img_prefix'] = os.path.join(DATA_ROOT, 'val2017/') 
    cfg['data']['test']['pipeline'].insert(1, aug)
    cfg['evaluation']['jsonfile_prefix'] = 'output_failure_testset'
    cfg.dump(new_cfg_file)

def setup(models, prefix):
    runsh_str = '\n'
    for model in models:
        # tmp
        if model["Name"] in [prefix+'_r101_fpn_1x_coco', prefix+'_r50_fpn_1x_coco']: continue
        if 'caffe' in model["Name"] or 'poly' in model["Name"]: continue
        print('processing',model['Name'])

        base_dir = os.path.join(WORKSPACEROOT, prefix, model['Name'])
        os.makedirs(base_dir, exist_ok=True)
        #os.system(f'wget -P {base_dir} {model["Weights"]}')
        test_base_dir = os.path.join(TEST_BASE_ROOT, prefix, model['Name'])
        weight_path = os.path.join(test_base_dir, model["Weights"].split('/')[-1])

        clean = get_clean(test_base_dir)
        if clean is None:
            continue

        for corruption in get_corruption_names():
            if corruption == 'glass_blur': continue
            for severity in [3]:
            #for severity in [1,2,4,5]:
            #tmp
            #for severity in range(1, 6):
                k = f'{corruption}-{severity}'
                print(k)

                rd = os.path.join(base_dir, k)
                os.makedirs(rd, exist_ok=True)
                
                # parse failure
                td = os.path.join(test_base_dir, k)
                fail_ann_path = os.path.join(td, FAILURE_ANN_NAME)
                if not os.path.exists(fail_ann_path):
                    fail_path = os.path.join(td, TEST_IMG_AP_C_FILE_NAME)
                    failure_imgid = parse_failure(fail_path, clean)
                    setup_failure_set(failure_imgid, td)

                new_cfg_file = os.path.join(rd, model["Config"].split('/')[-1]).replace('.py','_failure_testset.py')

                if not os.path.exists(new_cfg_file):
                    setup_cfg(model, corruption, severity, new_cfg_file, td, weight_path)

                    for sh in ['test2.sh']:
                        shutil.copy(sh, rd)

                runsh_str += f'cd {SET2RUN(rd)}\n'

                runsh_str+=f'sh test2.sh {SET2RUN(new_cfg_file)} {weight_path} $1\n'

        with open(os.path.join(WORKSPACEROOT, prefix, 'run_tmp.sh'), 'w') as fw:
            fw.write(runsh_str)


    with open(os.path.join(WORKSPACEROOT, prefix, 'run_fail.sh'), 'w') as fw:
        fw.write(runsh_str)


if __name__ == '__main__':
    for prefix in ['retinanet','faster_rcnn']:
    #for prefix in ['retinanet']:
        models = get_model_files(os.path.join(CFG_ROOT, f'configs/{prefix}/metafile.yml'))
        setup(models, prefix=prefix)

