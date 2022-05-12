import os

def get_test_new(dn, failed_id):
    fw0 = open(os.path.join(dn, 'test_all3.sh'), 'w')
    with open(os.path.join(dn, 'test_all2.sh')) as fr:
        for line in fr.readlines():
            if 'seq' in line and '#' not in line:
                line = 'for e in '+' '.join([str(_) for _ in failed_id])+'\n'
            fw0.write(line)
        fw0.close()

fw = open('mean.sh','w')
fwt = open('test_again.sh','w')
with open('skip.log') as fr:
    for line in fr.readlines():
        d = line.replace('skip','cd')
        dn = line.replace('skip ','').replace('\n','')
        sumlog = os.path.join(dn, 'sum.log')
        if os.path.exists(sumlog):
            with open(sumlog) as fr:
                for line in fr.readlines():
                    continue
                if line.startswith('epoch'):
                    continue
        fw.write(d)
        f = []
        for i in range(8):
            with open(os.path.join(dn, f'test_{i+1}.log')) as fr:
                for line in fr.readlines(): continue
                if not line.startswith('Ordered'):
                    f.append(i+1)
        print(dn, f)
        if len(f) > 0:
            get_test_new(dn, f)
            fwt.write(d)
            cfg = dn.split('/')[-3]+'.py'
            fwt.write(f'bash test_all3.sh {cfg}\n')

        fw.write(f'echo {d}\n')
        fw.write('python get_mean.py 1 > sum.log\ntail -n 2 sum.log\n')
