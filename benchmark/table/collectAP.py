import pickle
import os
from collections import OrderedDict
import imagecorruptions

AP_KEYS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']

def parse_corruption_ap(path):
    data = pickle.load(open(path,'rb'))
    corrupt = list(data.keys())[0] 
    severity = list(data[corrupt].keys())[0] 
    ap = data[corrupt][severity]['bbox']
    ap = {_:ap[_] for _ in AP_KEYS}

    return corrupt, severity, ap

def parse_clean_ap(path):
    with open(path) as fr:
        raw_ap = eval([_ for _ in fr.readlines()][-1])
        ap = {k.replace('bbox_m','').replace('_',''):v for k,v in raw_ap.items() if 'copypaste' not in k}
        ap = {_:ap[_] for _ in AP_KEYS}
        
    return ap

def get_model_results(model_workspace_dir):
    results = {}
    for root, d, files in os.walk(model_workspace_dir):
        for f in files:
            p = os.path.join(root, f)
            if f == 'output_results.pkl':
                print(root.split('/')[-2:])
                c, s, ap = parse_corruption_ap(p)
                m = root.split('/')[-2]
                if m not in results:
                    results[m] = {c:{s:ap}}
                elif c not in results[m]:
                    results[m][c] = {s:ap}
                else:
                    assert s not in results[m][c], f'{m} {c} {s}'
                    results[m][c][s] = ap
                    
            elif f == 'test.log' and 'clean' in p:
                print(root.split('/')[-2:])
                try:
                    ap = parse_clean_ap(p)
                except Exception as e:
                    print(e)
                    print('-'*20)
                    continue
                m = root.split('/')[-2]
                if m not in results:
                    results[m] = {'clean':ap}
                else:
                    results[m]['clean'] = ap
    return results 

def _ap_str(ap, k):
    if ap is None:
        return '&'
    else:
        return f'&{ap[k]*100:.1f}'


if __name__=='__main__':
    results = get_model_results('../retinanet')

    from collectFailure import output_table
    outs = ''
    for k in AP_KEYS:
        _ap_str = lambda ap: f'&{ap[k]*100:.1f}' if ap is not None else '&'
        outs += output_table(results, str(k), _ap_str, True)

    with open('table-retina-2.txt','w') as fw:
        fw.write(outs)

