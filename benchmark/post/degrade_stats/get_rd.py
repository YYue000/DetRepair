from collect.collectAP import get_model_results
import numpy as np
import scipy.stats as stats


def get_rDG(minfo, key, r=True, rt_all=False):
    mean_d = 0.0
    mpcs = {}
    cleanap = 0

    for corruption, c_info in minfo.items():
        c_mpc = 0.0
        if corruption == 'clean':
            cleanap = c_info[key]
            continue
        for s, ap in c_info.items():
            c_mpc += ap[key]
        mpcs[corruption] = c_mpc / len(c_info)

    if rt_all:
        _mpcs = {c:cleanap-_ for c,_ in mpcs.items()}
        if r:
            _mpcs = {c:_/cleanap for c,_ in _mpcs.items()}
        return _mpcs
    mean_d = cleanap-sum(mpcs.values())/len(minfo)
    if r:
        mean_d = mean_d/cleanap
    return mean_d

def _output(l):
    l = dict(sorted(l.items(), key=lambda item: item[1]))
    for m, v in l.items():
        print(f'{m} {v:.3f}')

def _stat(cl):
    cll = {c:list(l.values()) for c,l in cl.items()}
    _output({c:np.mean(_) for c,_ in cll.items()})

    N = len(cll)
    kend_mt = np.zeros([N, N])
    Ks =list(cll.keys()) 

    s = ''
    for i in range(len(Ks)):
        s += Ks[i]+' '
    s+='\n'
    for i in range(len(Ks)):
        for j in range(len(Ks)):
            t, _ = stats.kendalltau(cll[Ks[i]], cll[Ks[j]])
            kend_mt[i][j] = t
            s+=f'{t:.3f} '
        s+='\n'

    print(s)
    print('-'*20)
    idx = np.unravel_index(np.argmin(kend_mt), kend_mt.shape)
    print('min',np.min(kend_mt), Ks[idx[0]], Ks[idx[1]])

    t_kend_mt = kend_mt.copy()
    x = np.arange(len(Ks))
    t_kend_mt[x,x] = -1
    idx = np.unravel_index(np.argmax(t_kend_mt), t_kend_mt.shape)
    print('max',np.max(t_kend_mt), Ks[idx[0]], Ks[idx[1]])

    

if __name__ == '__main__':
    results = get_model_results('../../retinanet')
    l = {}
    rt_all = True
    r = True
    for m,minfo in results.items():
        rpc = get_rDG(minfo, 'AP', r, rt_all)
        if rt_all:
            for c, v in rpc.items():
                if c not in l.keys():
                    l[c] = {}
                l[c][m] = v
        else:
            l[m] = rpc
        #print(f'{m} {rpc:.3f}')
    if not rt_all:
       _output(l)
    else:
        for c, _ in l.items():
            print(c,'-'*20)
            _output(_)

        _stat(l)

