import os
import yaml
import pickle
from collections import OrderedDict

from utils import output_table

def _get_fl_test_log(root):
    p = os.path.join(root, 'test2.log')
    if os.path.exists(p): return p
    p = os.path.join(root, 'test.log')
    if os.path.exists(p): return p
    print('no valid file', root)
    return None

def get_model_files(prefix):
    meta_file = f'/home/yueyuxin/mmdetection/configs/{prefix}/metafile.yml'
    res = yaml.safe_load(open(meta_file))
    return res['Models']

def parse_test_log_ap(path):
    with open(path) as fr:
        raw_ap = eval([_ for _ in fr.readlines()][-1])
        ap = {k.replace('bbox_m','').replace('_',''):v for k,v in raw_ap.items() if 'copypaste' not in k}
        #ap = {_:ap[_] for _ in AP_KEYS}
        ap = {'failure': ap['AP'], 'clean': None}
        
    return ap

def parse_sum_log_ap(path):
    with open(path) as fr:
        for l in fr.readlines(): continue
        sp = l.split(' ')
        mean_ap, fl_ap, cl_ap = float(sp[3]), float(sp[5]), float(sp[7])
    return {'mean': mean_ap, 'failure': fl_ap, 'clean': cl_ap}

def get_model_raw_failure_test_results(model_workspace_dir, verbose=True):
    meta_files = {pk: get_model_files(pk) for pk in ['retinanet', 'faster_rcnn']}
    clean_results = {}
    for pk in ['retinanet', 'faster_rcnn']:
        y = meta_files[pk]
        for m in ['r50_fpn_1x_coco', 'r101_fpn_1x_coco']:
            p = pk+'_'+m
            for it in y:
                if it['Name'] == p:
                    assert len(it['Results']) == 1
                    clean_results[p] = it['Results'][0]['Metrics']['box AP']/100.0


    results = {}

    assert os.path.exists(model_workspace_dir)
    for root, d, files in os.walk(model_workspace_dir):
        for f in files:
            if f == 'output_results.pkl': # whole val dataset
                p = _get_fl_test_log(root)
                m,_ = root.split('/')[-2:]
                sp = _.split('-')
                c = sp[0]
                s = int(sp[1])
 
                # tmp
                if not m.endswith('r50_fpn_1x_coco') and not m.endswith('r101_fpn_1x_coco'): continue
                if s != 3: continue
                if verbose:
                    print(root.split('/')[-2:])

                try:
                    ap = parse_test_log_ap(p)
                except Exception as e:
                    print(e)
                    print('-'*20)
                    continue

                ap['clean'] = clean_results[m]
                ap['mean'] = (ap['clean']+ap['failure'])/2.0
                if m not in results:
                    results[m] = {c:{s:ap}}
                elif c not in results[m]:
                    results[m][c] = {s:ap}
                else:
                    assert s not in results[m][c], f'{m} {c} {s}'
                    results[m][c][s] = ap

    return results 

def get_model_repair_results(repair_workspace_dir):
    assert os.path.exists(repair_workspace_dir)
    results = {}
    for root, d, files in os.walk(repair_workspace_dir):
        for f in files:
            if f == 'sum.log':
                model,_ = root.split('/')[-2:]
                sp = _.split('-')
                if len(sp) != 2: continue
                c = sp[0]
                s = int(sp[1])
                
                try:
                    ap = parse_sum_log_ap(os.path.join(root, f))
                except:
                    print('Failure:',root)
                    continue
                if model not in results:
                    results[model] = {c:{s:ap}}
                elif c not in results[model]:
                    results[model][c] = {s:ap}
                else:
                    assert s not in results[model][c], f'{model} {c} {s}'
                    results[model][c][s] = ap

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

                ap = {k:rpr_ap[k]-bsl_ap[k] for k in ['failure','clean','mean']}
                for k in ['failure','clean','mean']:
                    k2 = 'rel-'+k
                    ap[k2] = ap[k]/bsl_ap[k]

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


    table = ''
    for k in ['failure','clean','mean']:
    #for k in ['mean']:
        k2 = f"rel-{k}"
        table += output_table(results, 'repair-'+label+f'-{k}', lambda x: f'&{x[k2]:.3f}' if x is not None else '&')
    
    #with open(label+'.txt', 'w') as fw:
    #    fw.write(table)
    #pickle.dump(results, open(label+'.pkl','wb'))


