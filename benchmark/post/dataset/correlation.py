import numpy as np
import scipy.stats as stats

from collect.collectAP import get_model_results as get_ap_results
from collect.collectFailure import get_model_results as get_failure_results
from get_inst import get_failure_stats

from functools import partial
import json

def get_model_DG(minfo, r=True):
    mean_d = 0.0
    cleanap = 0
    res = {}

    for corruption, c_info in minfo.items():
        if corruption == 'clean':
            cleanap = c_info
            break
    for corruption, c_info in minfo.items():
        if corruption == 'clean':
            continue
        res[corruption] = {}
        for s, ap in c_info.items():
            res[corruption][s] = {}
            for key in AP_KEYS:
                if r:
                    res[corruption][s][key] = 1 - ap[key]/cleanap[key]
                else:
                    res[corruption][s][key] = cleanap[key]-ap[key]
    return res

def get_degradation(ap_results, r=True):
    results = {}
    for m, minfo in ap_results.items():
        results[m] = get_model_DG(minfo, r)
    return results

AP_KEYS = ['AP']
#AP_KEYS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']
areaRngLbl = ['all', 'small', 'medium', 'large']

if __name__ == '__main__':
    APITEM = 1 #0 for ap; 1 for degradation

    ann_file = '/yueyuxin/data/coco/annotations/instances_val2017.json'
    anns = json.load(open(ann_file))

    path = '../../'
    #path = '../../retinanet'
    
    raw_ap_results = get_ap_results(path)
    #degradation_results = get_degradation(raw_ap_results, r=False)
    degradation_results = get_degradation(raw_ap_results, r=True)
    if APITEM == 1:
        ap_results = degradation_results
    else:
        ap_results = raw_ap_results

    failure_results = get_failure_results(path, get_failure_func=partial(get_failure_stats, anns=anns))

    Object = 'area'

    if Object == 'area':
        L = lambda x: areaRngLbl 
        F = lambda fl_c_info,s,k,lbl: fl_c_info[s]['area_num'][lbl]/fl_c_info[s]['instance_num']
    else:
        raise NotImplementedError

    
    for k in AP_KEYS:
        for lbl in L(k):
        #for lbl in areaRngLbl:
        #for lbl in ['crowdratio']:
        #for lbl in [k]:
            ap = []
            fl = []

            for m in failure_results.keys():
                fl_info = failure_results[m]
                ap_info = ap_results[m]
                for c in fl_info.keys():
                    fl_c_info = fl_info[c]
                    ap_c_info = ap_info[c]
                    for s in fl_c_info.keys():
                        if fl_c_info[s] is None: continue
                        fl.append(F(fl_c_info,s,k,lbl))
                        #fl.append(fl_c_info[s]['crowd_num']/fl_c_info[s]['instance_num'])
                        #fl.append(fl_c_info[s]['area_num'][lbl]/fl_c_info[s]['instance_num'])
                        #fl.append(fl_c_info[s][k])
                        ap.append(ap_c_info[s][k])
            print([f'{_:.3f}' for _ in ap[:20]])
            print([f'{_:.3f}' for _ in fl[:20]])
            print(k,lbl,stats.kendalltau(ap, fl))
            print('-'*20)

