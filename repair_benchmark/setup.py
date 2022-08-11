import json
import yaml
import pickle
import shutil
import os
import copy

import numpy as np

from imagecorruptions import get_corruption_names
from mmcv import Config

MAXEXP = 5

#METHOD ='finetune_failure'  
#METHOD ='finetune_clean+failure' # 'finetune_failure'  
#METHOD ='finetune_calibrated' # 'finetune_failure'  
#METHOD = 'finetune_mosaic'
METHOD = 'finetune_pmosaic'

VAL_CLEAN = False

WEIGHTPATH = 'calibrated' if METHOD == 'finetune_calibrated' else 'download'
BN_CALIBRATION_MODEL_ROOT = '/yueyuxin/mmdetection/repair_benchmarks/bn-calibration/models'

MODELS = ['retinanet_r50_fpn_1x_coco', 'retinanet_r101_fpn_1x_coco', 'faster_rcnn_r50_fpn_1x_coco', 'faster_rcnn_r101_fpn_1x_coco',
'fcos_r50_caffe_fpn_gn-head_1x_coco', 'fcos_r101_caffe_fpn_gn-head_1x_coco',
'detr_r50_8x2_150e_coco']

CFG_ROOT = '/home/yueyuxin/mmdetection'
DATA_ROOT = '/yueyuxin/data/coco'
WORKSPACEROOT = '/yueyuxin/mmdetection/repair_benchmarks/'+METHOD

SET2RUN = lambda x: x#lambda x:x.replace('/home/yueyuxin','')
 
TEST_BASE_ROOT='/yueyuxin/mmdetection/corruption_benchmarks'
TEST_IMG_AP_C_FILE_NAME = 'output.pkl.ap.pkl'
TEST_IMG_AP_CLEAN_FILE_NAME = 'output.ap.pkl'
FAILURE_ANN_NAME = 'failure_annotation.json'
FL_TRAIN_NAME = lambda all_name, t: all_name.replace('.json',f'_fltrain_{t}.json')
FL_TEST_NAME = lambda all_name, t: all_name.replace('.json',f'_fltest_{t}.json')
#SAMPLE_CLEAN_TRAINING_PATH = '/yueyuxin/data/coco/sampled_annotations/instances_train2017_sample1000.json'
#FL_MG_TRAIN_NAME = FAILURE_ANN_NAME.replace('.json','_fltrain_cltrain.json')

def get_coco_annotations(path):
    return json.load(open(path))
coco_train_annotations = get_coco_annotations('/yueyuxin/data/coco/annotations/instances_train2017.json')
coco_val_annotations = get_coco_annotations('/yueyuxin/data/coco/annotations/instances_val2017.json')

MAXEPOCH = 24
################################################
################################################


def __get_last_line(filename):
    """
    get last line of a file
    :param filename: file name
    :return: last line or None for empty file
    """
    try:
        filesize = os.path.getsize(filename)
        if filesize < 100:
            return None
        else:
            with open(filename, 'rb') as fp: # to use seek from end, must use mode 'rb'
                offset = -8                 # initialize offset
                while -offset < filesize:   # offset cannot exceed file size
                    fp.seek(offset, 2)      # read # offset chars from eof(represent by number '2')
                    lines = fp.readlines()  # read from fp to eof
                    if len(lines) >= 2:     # if contains at least 2 lines
                        return lines[-1]    # then last line is totally included
                    else:
                        offset *= 2         # enlarge offset
                fp.seek(0)
                lines = fp.readlines()
                return lines[-1]
    except FileNotFoundError:
        print(filename + ' not found!')
        return None

def get_test_num(fp):
    l = []
    with open(fp) as fr:
        for line in fr.readlines():
            if 'Precision' in line and ' all' in line and ':' in line:
                line = line.replace('\n','')
                sp = line.split('=')
                ap = float(sp[-1])
                l.append(ap)
    return len(l)


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
    sample = np.random.choice(np.arange(len(annotations['images'])), samplesize, replace=False)
    annotations['images'] = [_ for _ in raw_annotations['images'] if _['id'] in sample]
    annotations['annotations'] = [_ for _ in raw_annotations['annotations'] if _['image_id'] in sample]
    return annotations

def merge_samples(base_annotations, app_ann_list, name, dump=True):
    for ann in app_ann_list:
        for k in ['images', 'annotations']:
            base_annotations[k] += app[k]
    if dump:
        json.dump(base_annotations, open(name, 'w'))

def setup_all_failure_set(failure_imgid, wksp):
    name = os.path.join(wksp, FAILURE_ANN_NAME)
    get_failure_set(coco_val_annotations, failure_imgid, name)

def setup_failure_train_test(failure_imgid, sample_names={}, sample_clean=False):
    train_name, test_name = sample_names['train'], sample_names['test']
    train, test = split_train_test(failure_imgid)

    train_ann = get_failure_set(coco_val_annotations, train, train_name)

    get_failure_set(coco_val_annotations, test, test_name)

    if sample_clean:
        raise NotImplementedError
        #clean = sample_train(coco_train_annotations)
        clean = get_coco_annotations(SAMPLE_CLEAN_TRAINING_PATH)
        mg_name = os.path.join(wksp, FL_MG_TRAIN_NAME)
        merge_samples(train_ann, [clean], mg_name)


def setup_cfg(model, corruption, severity, new_cfg_file, weight_path, train_ann_path, test_ann_path):
    cfg_path = os.path.join(CFG_ROOT, model["Config"])  
    cfg = Config.fromfile(cfg_path)
    cfg['load_from'] = SET2RUN(weight_path)
    train_ann_path = SET2RUN(train_ann_path)
    test_ann_path = SET2RUN(test_ann_path)

    aug = dict(type='Corrupt',corruption=corruption, severity=severity)
    train_data = cfg['data']['train']
    if train_data['type'] == 'CocoDataset':
        pass
    elif train_data['type'] == 'RepeatDataset':
        train_data = train_data['dataset']
    else:
        print(train_data['type'])
        raise NotImplementedError
    train_data['ann_file'] = train_ann_path
    train_data['img_prefix'] = os.path.join(DATA_ROOT, 'val2017/') 

    if METHOD in ['finetune_mixup', 'finetune_mosaic', 'finetune_pmosaic']:
        train_pipeline_local = train_data['pipeline'][:2]
        train_pipeline_global = train_data['pipeline'][2:]
        train_data['pipeline'] = train_pipeline_local

        if METHOD == 'finetune_mixup':
            mixup = dict(type='MixUp', img_scale=(1024,1024), ratio_range=(0.8,1.6))    
            train_pipeline_global.insert(1, mixup)
        elif METHOD == 'finetune_pmosaic':
            mosaic = dict(type='PMosaic', prob=0.5, img_scale=(1024,1024), pad_val=0)
            train_pipeline_global.insert(0, mosaic)
        elif METHOD == 'finetune_mosaic':
            mosaic = dict(type='Mosaic', img_scale=(1024,1024), pad_val=0)
            train_pipeline_global.insert(0, mosaic)

    corrupted_train_data = copy.deepcopy(train_data)
    corrupted_train_data['pipeline'].insert(1, aug)

    if METHOD in ['finetune_clean+failure','finetune_calibrated']:
        new_train_data = dict(type='ConcatDataset', datasets=[train_data, corrupted_train_data])
    elif METHOD == 'finetune_failure':
        new_train_data = corrupted_train_data
    elif METHOD in ['finetune_mixup', 'finetune_mosaic', 'finetune_pmosaic']:
        _train_data = dict(type='ConcatDataset', datasets=[train_data, corrupted_train_data])
        new_train_data = dict(type='MultiImageMixDataset', dataset=_train_data, pipeline=train_pipeline_global)
    elif METHOD == 'finetune_augmix':
        new_train_data = copy.deepcopy(train_data)
        new_train_data['pipeline'].insert(1, dict(type='FakeAugMix', corruption=corruption, severity=severity, width=2))
    else:
        raise NotImplementedError

    assert cfg['data']['val']['type'] == 'CocoDataset'
    assert cfg['data']['test']['type'] == 'CocoDataset'
    cfg['data']['val']['ann_file'] = test_ann_path
    cfg['data']['val']['img_prefix'] = os.path.join(DATA_ROOT, 'val2017/') 
    
    cfg['data']['test']['ann_file'] = test_ann_path
    cfg['data']['test']['img_prefix'] = os.path.join(DATA_ROOT, 'val2017/') 
    cfg['data']['test']['pipeline'].insert(1, aug)
    if not VAL_CLEAN:
        test_data = cfg['data']['test']
        clean_test_data = copy.deepcopy(test_data)
        clean_test_data['pipeline'].pop(1)
        cfg['data']['test'] = dict(type='ConcatDataset', datasets=[test_data, clean_test_data])
        cfg['evaluation']['interval'] = MAXEPOCH+1
    cfg['data']['train'] = new_train_data
    # if batch=8
    cfg['data']['samples_per_gpu'] = 8
    
    cfg['runner']['max_epochs'] = MAXEPOCH
    if cfg['lr_config']['policy'] == 'step':
        cfg['optimizer']['lr'] /= np.power(10.0, len(cfg['lr_config']['step']))
        cfg['lr_config']['step'] = [MAXEPOCH+1]
    else:
        print(cfg['lr_config'])
        raise NotImplementedError
    cfg['lr_config'].pop('warmup', None)
    cfg['lr_config'].pop('warmup_iters', None)
    cfg['lr_config'].pop('warmup_ratio', None)

    cfg.dump(new_cfg_file)

def setup(models, prefix):
    trainsh_str = ''
    testsh_str = ''
    for model in models:
        if model['Name'] not in MODELS: continue
        #tmp
        if model['Name'] != 'retinanet_r101_fpn_1x_coco': continue
        print('processing',model['Name'])

        base_dir = os.path.join(WORKSPACEROOT, prefix, model['Name'])
        os.makedirs(base_dir, exist_ok=True)
        #os.system(f'wget -P {base_dir} {model["Weights"]}')
        test_base_dir = os.path.join(TEST_BASE_ROOT, prefix, model['Name'])
        if WEIGHTPATH == 'download':
            weight_path = lambda c: os.path.join(test_base_dir, model["Weights"].split('/')[-1])
        elif WEIGHTPATH == 'calibrated':
            weight_path = lambda c: os.path.join(BN_CALIBRATION_MODEL_ROOT, model["Weights"].split('/')[-1].replace('.pth', f'-{c}-best.pth'))

        clean = get_clean(test_base_dir)
        if clean is None:
            continue

        corruption_list = list(get_corruption_names())
        corruption_list.remove('glass_blur')
        corruption_list.append('glass_blur')

        for corruption in corruption_list:
            for severity in [3]:
            #for severity in range(1, 6):
                k = f'{corruption}-{severity}'
                print(k)
                # parse failure
                td = os.path.join(test_base_dir, k)
                all_fail_ann_path = os.path.join(td, FAILURE_ANN_NAME)
                if not os.path.exists(all_fail_ann_path):
                    fail_path = os.path.join(td, TEST_IMG_AP_C_FILE_NAME)
                    failure_imgid = parse_failure(fail_path, clean)
                    setup_all_failure_set(failure_imgid, td)
                else:
                    fail_ann = get_coco_annotations(all_fail_ann_path)
                    failure_imgid = [_['id'] for _ in fail_ann['images']]

                for t in range(MAXEXP):
                    rd = os.path.join(base_dir, k, f'exp{t}')
                    
                    # trained exp
                    train_ok = False
                    ckpt_path = os.path.join(rd, 'work_dirs', f'epoch_{MAXEPOCH}.pth')
                    if os.path.exists(ckpt_path): 
                        print(f'skip {rd} in train')
                        train_ok = True

                    # finished exp
                    sum_log_path = os.path.join(rd, 'sum.log')
                    if os.path.exists(sum_log_path):
                        line = __get_last_line(sum_log_path)
                        if line is not None:
                            line = str(line)
                            if line.startswith('epoch '):
                                try:
                                    _e_maxep = int(line.replace('\n','').split('in')[1]) 
                                    if _e_maxep == MAXEPOCH:
                                        print(f'skip {rd} in test')
                                        continue
                                    else:
                                        print('epoch error')
                                except Exception as e:
                                    print(e, 'at epoch',corruption,t)
                        os.remove(sum_log_path)
                    elif train_ok:
                        test_num = get_test_num(os.path.join(rd, 'test.log'))
                        if VAL_CLEAN and test_num == MAXEPOCH or (not VAL_CLEAN) and test_num==MAXEPOCH*2:
                            print('test ok but no sum log', rd)
                            continue
                        elif test_num>0:
                            print(f'testing, {test_num} tests got')
                            continue



                    new_cfg_file = os.path.join(rd, model["Config"].split('/')[-1])

                    if not train_ok:
                        os.makedirs(rd, exist_ok=True)

                        fail_ann_path_train = FL_TRAIN_NAME(all_fail_ann_path, t)
                        fail_ann_path_test = FL_TEST_NAME(all_fail_ann_path, t)
                        sampled_ann_paths = {'train': fail_ann_path_train, 'test':fail_ann_path_test}
                        if not os.path.exists(fail_ann_path_train) or not os.path.exists(fail_ann_path_test):
                            setup_failure_train_test(failure_imgid, sampled_ann_paths)
                            print(f'sampling {fail_ann_path_train}')
                        


                        if not os.path.exists(new_cfg_file):
                            setup_cfg(model, corruption, severity, new_cfg_file, weight_path(corruption), fail_ann_path_train, fail_ann_path_test)
                            for sh in ['train.sh', 'test_all.sh', 'get_mean.py']:
                                shutil.copy(sh, rd)

                        trainsh_str += f'cd {SET2RUN(rd)}\n'
                        trainsh_str+=f'bash train.sh {SET2RUN(new_cfg_file)} $1\n'
                    
                    testsh_str += f'cd {SET2RUN(rd)}\n'

                    testsh_str+=f'sh test_all.sh {SET2RUN(new_cfg_file)} {MAXEPOCH} $1\n'
                    if VAL_CLEAN:
                        testsh_str += 'python get_mean.py 2>&1|tee sum.log\n'
                    else:
                        testsh_str += f'python get_mean.py --test_in 2 --MAXEPOCH {MAXEPOCH} 2>&1|tee sum.log\n'


        #with open(os.path.join(WORKSPACEROOT, prefix, 'run_tmp.sh'), 'w') as fw:
            #fw.write(runsh_str)


    with open(os.path.join(WORKSPACEROOT, prefix, 'run_test_r101_2.sh'), 'w') as fw:
        fw.write(testsh_str)

    with open(os.path.join(WORKSPACEROOT, prefix, 'run_train_r101_2.sh'), 'w') as fw:
        fw.write(trainsh_str)



if __name__ == '__main__':
    #for prefix in ['retinanet','faster_rcnn', 'detr', 'fcos']:
    for prefix in ['retinanet']:
        models = get_model_files(os.path.join(CFG_ROOT, f'configs/{prefix}/metafile.yml'))
        setup(models, prefix=prefix)

