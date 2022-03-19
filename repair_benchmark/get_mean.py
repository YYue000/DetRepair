import os
import numpy as np

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

clean = get_ap('train.log')
fog5 = get_ap('test.log')

l = [(_+__)/2 for _,__ in zip(clean, fog5)]
print(l)
_ = np.argmax(l)
print("epoch",_+1,"mean",max(l), "failure",fog5[_], "clean",clean[_])
