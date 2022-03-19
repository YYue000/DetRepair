import numpy as np
import scipy.stats as stats

from collect.collectAP import get_model_results as get_ap_results
from collect.collectFailure import get_model_results as get_failure_results
from get_inst import get_failure_stats
from degrade_stats.get_rd import get_rDG

from functools import partial
import json

AP_KEYS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']
areaRngLbl = ['all', 'small', 'medium', 'large']

if __name__ == '__main__':
    ann_file = '/yueyuxin/data/coco/annotations/instances_val2017.json'
    anns = json.load(open(ann_file))


    path = '../../retinanet'
    ap_results = get_ap_results(path)
    failure_results = get_failure_results(path, get_failure_func=partial(get_failure_stats, anns=anns))


    for k in AP_KEYS:
        #for lbl in areaRngLbl:
        for lbl in ['crowdratio']:
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
                        fl.append(fl_c_info[s]['crowd_num']/fl_c_info[s]['instance_num'])
                        #fl.append(fl_c_info[s]['area_num'][lbl]/fl_c_info[s]['instance_num'])
                        ap.append(ap_c_info[s][k])
            #print(ap[:20])
            #print(fl[:20])
            print(k,lbl,stats.kendalltau(ap, fl))

