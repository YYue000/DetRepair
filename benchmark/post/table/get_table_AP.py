import pickle
import os
from collections import OrderedDict
from collect.collectAP import get_model_results

AP_KEYS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']

if __name__=='__main__':
    results = get_model_results('../../retinanet')
    print(len(results))

    from utils import output_table
    outs = ''
    for k in AP_KEYS:
        _ap_str = lambda ap: f'&{ap[k]*100:.1f}' if ap is not None else '&'
        outs += output_table(results, str(k), _ap_str, True)

    with open('table-retina-2.txt','w') as fw:
        fw.write(outs)

