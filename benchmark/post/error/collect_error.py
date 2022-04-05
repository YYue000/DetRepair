import pickle
import os

AREALBL = ['all', 'small', 'medium', 'large']

def _parse_file(p):
    res = {k:{} for k in AREALBL}
    with open(p) as fr:
        for line in fr.readlines():
            line = line.replace('\n','')
            sp = line.split(' ')
            area = sp[0][:-1]
            ap = eval(sp[1])[0]
            name = sp[2]
            res[area][name] = ap
    return res

def _get_error(res):
    error = {}
    for k, it in res.items():
        e = {}
        e['fne'] = res[k]['FN']-res[k]['BG']
        e['bge'] = res[k]['BG']-res[k]['Oth']
        e['loce'] = res[k]['Loc']-res[k]['C50']
        e['clse'] = res[k]['Oth'] - res[k]['Loc']

        error[k] = e
    return error

def cmp_error(Errors):
    res = {}
    for area in AREALBL:
        res[area] = {}
        for model, meinfo in Errors.items():
            clean = None
            res[area][model] = {}
            for c_s, ap in meinfo.items():
                if c_s == 'clean':
                    clean = ap[area]
                    break
            for c_s, ap in meinfo.items():
                if c_s == 'clean': continue
                _ap = ap[area]
                md = -1
                mdk = None
                for k,v in _ap.items():
                    d = (clean[k]-_ap[k])/clean[k]
                    if d>md:
                        md = d
                        mdk = k
                res[area][model][c_s] = mdk
                print(area,model,c_s,mdk)
    return res

def get_Errors(path, dump_file=None):
    Errors = {}
    for root, d, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if not p.endswith('analyze/ls.txt'): continue
            res = _parse_file(p)
            error = _get_error(res)

            model, c_s = root.split('/')[-3:-1]
            print(model, c_s, error)
            if model not in Errors:
                Errors[model] = {}
            Errors[model][c_s] = error
    
    if dump_file:
        pickle.dump(Errors, open(dump_file,'wb'))
    return Errors


if __name__ == '__main__':
    path = '/home/yueyuxin/repair_workspace/mmdetection/corruption_benchmarks/retinanet/'
    #dump_file = 'retinanet_r50_fpn_1x_coco_errors.pkl'
    dump_file = 'retinanet_errors.pkl'
    Errors = get_Errors(path, dump_file)
    
    maxerror_keys = cmp_error(Errors)
    pickle.dump(maxerror_keys, open('retinanet_maxerror_keys.pkl', 'wb'))
