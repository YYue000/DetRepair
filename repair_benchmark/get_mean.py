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

def get_concat_ap(fp, MAXEPOCH):
    cl = np.zeros(MAXEPOCH)
    fl = np.zeros(MAXEPOCH)
    with open(fp) as fr:
        e = 0
        for line in fr.readlines():
            if 'epoch_' in line:
                e = int(line.replace('\n','').split('_')[-1].split('.pth')[0])
            if 'copypaste' in line:
                line = line.replace('\n','')
                ap = eval(line)
                ap0 = ap['0_bbox_mAP']
                ap1 = ap['1_bbox_mAP']
                if cl[e-1]!=0:
                    print('repeat at epoch',e)
                cl[e-1] = ap1
                fl[e-1] = ap0
                print(line,e,ap0, ap1)
        lost = [str(_+1) for _ in np.where(cl==0)[0]]
        assert len(lost) == 0, ' '.join(lost)
    return cl, fl


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_in', type=int, default=0)
    parser.add_argument('--MAXEPOCH', type=int, default=8)
    args = parser.parse_args()

    if args.test_in == 2:
        clean, fog5 = get_concat_ap('test.log', args.MAXEPOCH)
    else:
        clean = get_ap('train.log')
    assert len(clean) == args.MAXEPOCH


    if args.test_in == 0:
        fog5 = get_ap('test.log')
    elif args.test_in == 1:
        fog5 = [get_ap(f'test_{_+1}.log')[0] for _ in range(args.MAXEPOCH)]
    elif args.test_in == 2:
        pass
    else:
        raise NotImplementedError

    assert len(clean) == len(fog5)
    l = [(_+__)/2 for _,__ in zip(clean, fog5)]
    print(l)
    _ = np.argmax(l)
    print("epoch",_+1,"mean",max(l), "failure",fog5[_], "clean",clean[_],'in',args.MAXEPOCH)
