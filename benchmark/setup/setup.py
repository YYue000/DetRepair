import yaml
import shutil
import os
from imagecorruptions import get_corruption_names
    
TEST_SH = {'clean': 'test_scripts/test.sh'}

for corruption in get_corruption_names():
    for serverity in [3]:
    #for serverity in range(1, 6):
        TEST_SH[f'{corruption}-{serverity}'] = f'test_scripts/test_{corruption}-{serverity}.sh'

def _get_cfg(seg):
    return seg.split('/')[-1][:-1]

def _get_pth(seg):
    p = seg.split(')')[0]
    return p[p.find('(')+1:]

def get_model_files(meta_file):
    res = yaml.safe_load(open(meta_file))
    return res['Models']

def get_model_files_fake(readme_file):
    models = []
    flag = 0
    with open(readme_file) as fr:
        for line in fr.readlines():
            if 'Results and Models' in line:
                flag = 1
                continue
            if flag > 0 and '----' in line:
                flag = 2
                continue
            if flag == 2:
                sp = line.replace('\n','').split('|')
                cfg = _get_cfg(sp[-2])
                pth_path = _get_pth(sp[-1])
                models.append({'cfg': cfg, 'pth': pth_path})
            if flag == 2 and line[0] == '#':
                break
    return models

def setup(models, root, cfg_dir, prefix):
    runsh_str = '\n'
    for model in models:
        if 'caffe' in model['Name'] or 'poly' in model['Name']: continue
        base_dir = os.path.join(root, prefix, model['Name'])
        print('processing',model['Name'])
        try:
            w = model["Weights"].split('/')[-1]
        except:
            print('No weights available', model['Name'])
            continue

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)   
            os.system(f'wget -P {base_dir} {model["Weights"]}')
        for k, SH in TEST_SH.items():
            d = os.path.join(base_dir, k)
            if not os.path.exists(d):
                os.makedirs(d)
                shutil.copy(SH, d)

            runsh_str += f'cd {d}\n'
            runsh_str+=f'sh {SH.split("/")[-1]} {os.path.join(cfg_dir, model["Config"])} ../{w} $1\n'

    with open(os.path.join(root, prefix,'run.sh'), 'w') as fw:
        fw.write(runsh_str)


if __name__ == '__main__':
    cfg_dir = '/home/yueyuxin/mmdetection'
    prefix = 'mask_rcnn'
    models = get_model_files(os.path.join(cfg_dir, f'configs/{prefix}/metafile.yml'))
    setup(models, '/yueyuxin/mmdetection/corruption_benchmarks', cfg_dir, prefix=prefix)

