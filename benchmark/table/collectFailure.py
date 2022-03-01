import pickle
import os
import numpy as np
from utils import output_table

def get_clean(result_file):
    return pickle.load(open(result_file, 'rb'))

def get_failure(result_file, clean, B=1, key_idx=1):
    results = pickle.load(open(result_file, 'rb'))
    failure_flag = results[key_idx]<clean[key_idx]
    failure_imgid = results[0][failure_flag]
    failure_pctg = np.sum(failure_flag)/results.shape[1]
    return failure_pctg


def parse_failure(root, name='output.pkl.ap.pkl', clean_name='output.ap.pkl'):
    sp = root.split('/')
    model = sp[-2]
    ssp = sp[-1].split('-')
    corruption, severity = ssp[0], int(ssp[1])
    path = os.path.join(root, name)
    clean_path = os.path.join(*(sp[:-1]), 'clean', clean_name)
    if not os.path.exists(path) or not os.path.exists(clean_path):
        return corruption, severity, None
    clean = get_clean(clean_path)
    failure_pctg = get_failure(path, clean)
    return corruption, severity, failure_pctg


def get_model_results(model_workspace_dir):
    results = {}
    for root, d, files in os.walk(model_workspace_dir):
        for f in files:
            p = os.path.join(root, f)
            if f == 'output_results.pkl':
                print(root)
                c, s, fl = parse_failure(root)
                m = root.split('/')[-2]
                if m not in results:
                    results[m] = {c:{s:fl}}
                elif c not in results[m]:
                    results[m][c] = {s:fl}
                else:
                    assert s not in results[m][c], f'{m} {c} {s}'
                    results[m][c][s] = fl
    return results 

if __name__=='__main__':
    results = get_model_results('../retinanet')
    _ap_str = lambda ap: f'&{ap*100:.1f}' if ap is not None else '&'
    outs = output_table(results, 'failure', _ap_str)

    output_file = 'table-retina-failure.txt'
    with open(output_file, 'w') as fw:
        fw.write(outs)


