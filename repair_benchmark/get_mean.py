import os
import numpy as np
import pickle
import sys

def get_ap(fp):
    l = []
    with open(fp) as fr:
        for line in fr.readlines():
            if 'Precision' in line and ' all' in line and ':' in line:
                line = line.replace('\n','')
                sp = line.split('=')
                ap = float(sp[-1])
                l.append(ap)
                print(line,len(l),ap)
    return l

if __name__ == '__main__':
    MAXEPOCH = 8

    if len(sys.argv) == 1:
        test_in = 0
    else:
        test_in = int(sys.argv[1])

    clean = get_ap('train.log')
    assert len(clean) == MAXEPOCH

    if test_in == 0:
        fog5 = get_ap('test.log')
    elif test_in == 1:
        fog5 = [get_ap(f'test_{_}.log')[0] for _ in range(MAXEPOCH)]
    else:
        raise NotImplementedError

    l = [(_+__)/2 for _,__ in zip(clean, fog5)]
    print(l)
    _ = np.argmax(l)
    print("epoch",_+1,"mean",max(l), "failure",fog5[_], "clean",clean[_])
