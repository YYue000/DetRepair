import os
import yaml
import pickle
import numpy as np
from collections import OrderedDict

from utils import output_table

MODELS = ['retinanet_r50_fpn_1x_coco', 'retinanet_r101_fpn_1x_coco', 'faster_rcnn_r50_fpn_1x_coco', 'faster_rcnn_r101_fpn_1x_coco',
'fcos_r50_caffe_fpn_gn-head_1x_coco', 'fcos_r101_caffe_fpn_gn-head_1x_coco',
'detr_r50_8x2_150e_coco']

def _get_fl_test_log(root):
    p = os.path.join(root, 'test2.log')
    if os.path.exists(p): return p
    p = os.path.join(root, 'test.log')
    if os.path.exists(p): return p
    print('no valid file', root)
    return None

def _get_fl_test_log_clean(root):
    p = os.path.join(root, 'test_clean.log')
    if os.path.exists(p): return p
    print('no valid file', root)
    return None

def get_model_files(prefix):
    meta_file = f'/home/yueyuxin/mmdetection/configs/{prefix}/metafile.yml'
    res = yaml.safe_load(open(meta_file))
    return res['Models']

def parse_test_log_ap(path, k='failure'):
    with open(path) as fr:
        raw_ap = eval([_ for _ in fr.readlines()][-1])
        ap = {k.replace('bbox_m','').replace('_',''):v for k,v in raw_ap.items() if 'copypaste' not in k}
        #ap = {_:ap[_] for _ in AP_KEYS}
        ap = {k: ap['AP']}
        
    return ap

def parse_test_results(root):
    p = _get_fl_test_log(root)
    pc = _get_fl_test_log_clean(root)
 
    ap = parse_test_log_ap(p)
    apc = parse_test_log_ap(pc, 'clean')
    ap.update(apc)
    ap['mean'] = (ap['failure']+ap['clean'])/2
    return ap


def parse_sum_log_ap(path):
    with open(path) as fr:
        for l in fr.readlines(): continue
        sp = l.split(' ')
        mean_ap, fl_ap, cl_ap = float(sp[3]), float(sp[5]), float(sp[7])
    return {'mean': mean_ap, 'failure': fl_ap, 'clean': cl_ap}

def get_model_raw_failure_test_results(model_workspace_dir, verbose=True):
    results = {}

    assert os.path.exists(model_workspace_dir)
    for root, d, files in os.walk(model_workspace_dir):
        for f in files:
            if f == 'output_failure_testset.bbox.json':
            #if f == 'output_results.pkl': # whole val dataset
                rsp = root.split('/')
                if rsp[-1][:3]=='exp':
                    exp = int(rsp[-1].replace('exp',''))
                    model,_ = rsp[-3:-1]
                else:
                    exp = None
                    model,_ = rsp[-2:]
                if model not in MODELS: continue

                sp = _.split('-')
                if len(sp) != 2: continue
                c = sp[0]
                s = int(sp[1])

                if s != 3: continue

                if verbose:
                    print(root.split('/')[-2:])

                try:
                    ap = parse_test_results(root)
                except Exception as e:
                    print(e)
                    print('-'*20)
                    continue
                
                if model not in results:
                    results[model] = {}
                if c not in results[model]:
                    results[model][c] = {}
                
                assert s not in results[model][c], f'{model} {c} {s}'
                results[model][c][s] = {}
                assert exp not in results[model][c][s], f'{model} {c} {s} {exp}'
                results[model][c][s][exp] = ap


    return results 

def get_model_repair_results(repair_workspace_dir):
    assert os.path.exists(repair_workspace_dir)
    results = {}
    for root, d, files in os.walk(repair_workspace_dir):
        for f in files:
            if f == 'sum.log':
                rsp = root.split('/')
                if rsp[-1][:3]=='exp':
                    exp = int(rsp[-1].replace('exp',''))
                    model,_ = rsp[-3:-1]
                else:
                    exp = None
                    model,_ = rsp[-2:]
                sp = _.split('-')
                if len(sp) != 2: continue
                c = sp[0]
                s = int(sp[1])
                
                try:
                    ap = parse_sum_log_ap(os.path.join(root, f))
                except Exception as e:
                    print('Failure:',root,'of',e)
                    print(os.path.join(root,f))
                    continue
                
                if model not in results:
                    results[model] = {}
                if c not in results[model]:
                    results[model][c] = {}
                
                assert s not in results[model][c], f'{model} {c} {s}'
                results[model][c][s] = {}
                assert exp not in results[model][c][s], f'{model} {c} {s} {exp}'
                results[model][c][s][exp] = ap

    return results

if __name__ == '__main__':
    #repair_workspace = '/yueyuxin/tmp/repair_benchmarks/finetune_clean+failure'
    label = 'finetune_clean+failure'
    #label = 'finetune_failure'
    repair_workspace = f'../{label}/'
    bsl = get_model_raw_failure_test_results('../../corruption_benchmarks')
    rpr = get_model_repair_results(repair_workspace)

    results = {}

    for m, rpr_minfo in rpr.items():
        print(m)
        for c, rpr_cinfo in rpr_minfo.items():
            for s, rpr_ap in rpr_cinfo.items():
                bsl_ap = bsl[m][c][s]
                ap = {}
                for k in ['failure','clean','mean']:
                    assert len(rpr_ap)>=4, f'{m} {c} {len(rpr_ap[k])}'
                    l = list(rpr_ap.keys()) 
                    ap[k] = [rpr_ap[exp][k]-bsl_ap[exp][k] for exp in l]
                    k2 = 'rel-'+k
                    ap[k2] = [(rpr_ap[exp][k]-bsl_ap[exp][k])/bsl_ap[k] for exp in l]

                if m not in results:
                    results[m] = {c:{s:ap}}
                elif c not in results[m]:
                    results[m][c] = {s:ap}
                else:
                    assert s not in results[m][c], f'{m} {c} {s}'
                    results[m][c][s] = ap
        print(results[m].keys())
        _l = []
        for k, cinfo in results[m].items():
            for s in cinfo.keys():
                _l.append(f'{k}-{s}')
        print(_l)


    def _ap_str(x):
        if x is None:
            return '&'
        if len(x) != 5:
            print(x)
        m = np.mean(x[k2])
        r = (np.max(x[k2])-np.min(x[k2]))/2
        return f'&${m:.3f} \pm {r:.3f}$'
    #_ap_str = lambda x: f'&{x[k2]:.3f}' if x is not None else '&'
    table = ''
    for k in ['failure','clean','mean']:
    #for k in ['mean']:
        k2 = f"rel-{k}"
        table += output_table(results, 'repair-'+label+f'-{k}', _ap_str)
    print(table)
    with open(label+'.txt', 'w') as fw:
        fw.write(table)
    pickle.dump(results, open(label+'.pkl','wb'))


