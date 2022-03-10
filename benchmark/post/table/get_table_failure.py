import pickle
import os
import numpy as np
from utils import output_table
from collect.collectFailure import get_model_results

if __name__=='__main__':
    results = get_model_results('../../retinanet')
    _ap_str = lambda ap: f'&{ap*100:.1f}' if ap is not None else '&'
    outs = output_table(results, 'failure', _ap_str)

    output_file = 'table-retina-failure.txt'
    with open(output_file, 'w') as fw:
        fw.write(outs)


