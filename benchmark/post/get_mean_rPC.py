from collect.collectAP import get_model_results
import xlwt


def get_rPC(minfo, key, mode='rel'):
    mpc = 0.0
    cleanap = 0

    for corruption, c_info in minfo.items():
        c_mpc = 0.0
        if corruption == 'clean':
            cleanap = c_info[key]
            continue
        for s, ap in c_info.items():
            c_mpc += ap[key]
        mpc += c_mpc / len(c_info)
    mpc = mpc/len(minfo)
    if cleanap == 0:
        return None
    print(cleanap)
    if mode == 'rel':
        rpc = mpc/cleanap
    elif mode == 'abs':
        rpc = mpc-cleanap
    else:
        raise NotImplementedError
    return rpc

def to_excel(ls, det):
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet('rPC')
    worksheet.write(0,0,'model')

    model2idx = {}
    maxidx = 1
    for m in list(ls.values())[0].keys():
        model2idx[m] = maxidx
        worksheet.write(maxidx, 0, m)
        maxidx += 1
   
    for i,(k,l) in enumerate(ls.items()):
        worksheet.write(0, i+1, k)
        for m, rpc in l.items(): 
            worksheet.write(model2idx[m], i+1, f'{rpc:.3f}')

    workbook.save(f'{det}_rpc.xls')

if __name__ == '__main__':
    results = get_model_results('../../faster_rcnn')
    #results = get_model_results('../../retinanet')

    ls = {}
    for k in ['AP']:
    #for k in ['AP', 'APs', 'APm', 'APl', 'AP50', 'AP75']:
        print(k)
        l = {}
        for m,minfo in results.items():
            #rpc = get_rPC(minfo, k)
            rpc = get_rPC(minfo, k, mode='abs')
            if rpc is None:
                print('no clean data at',m)
                continue
            l[m] = rpc
            print(f'{m} {rpc:.3f}')
        print('-'*25)
        l = dict(sorted(l.items(), key=lambda item: item[1], reverse=True))
        for m, v in l.items():
            print(f'{m} {v:.3f}')
        print('='*25)
        ls[k] = l 
     

    #to_excel(ls, 'faster_rcnn_mode_abs')
