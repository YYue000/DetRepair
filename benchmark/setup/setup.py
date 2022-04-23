import yaml
import shutil
import os
import wget
from imagecorruptions import get_corruption_names

    
TEST_SH = {'clean': 'test_scripts/test.sh'}

for corruption in get_corruption_names():
    if corruption == 'glass_blur': continue
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
        if  prefix in ['cascade_rcnn', 'retinanet', 'faster_rcnn', 'mask_rcnn']: 
            if 'poly' in model['Name']: continue
            if 'caffe' in model['Name']: continue
        flag = False
        for r in model['Results']:
            if r['Task'] == 'Object Detection':
                flag = True
                break
        if not flag:
            print(prefix,model['Name'])
            continue

        base_dir = os.path.join(root, prefix, model['Name'])
        print('processing',prefix,model['Name'])
        try:
            w = model["Weights"].split('/')[-1]
        except:
            print('No weights available', model['Name'])
            continue

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)   
        if not os.path.exists(os.path.join(base_dir, w)):
            print(base_dir, w, model["Weights"])
            filename = wget.download(model["Weights"], out=os.path.join(base_dir,w))
            #os.system(f'wget -P {base_dir} {model["Weights"]}')
        for k, SH in TEST_SH.items():
            d = os.path.join(base_dir, k)
            if k == 'clean':
                if os.path.exists(os.path.join(d,'output.bbox.json')):
                    continue
            else:
                if os.path.exists(os.path.join(d,'output_results.pkl')):
                    continue
            # tmp
            shutil.copy(SH, d)
            if not os.path.exists(d):
                os.makedirs(d)
                shutil.copy(SH, d)

            runsh_str += f'cd {d}\n'
            runsh_str+=f'sh {SH.split("/")[-1]} {os.path.join(cfg_dir, model["Config"])} ../{w} $1\n'
    if len(runsh_str) > 1:
        with open(os.path.join(root, prefix,'run.sh'), 'w') as fw:
            fw.write(runsh_str)


if __name__ == '__main__':
    cfg_dir = '/home/yueyuxin/mmdetection'
    root = cfg_dir+'/configs'

    #prefix = 'cascade_rcnn'
    for prefix in ['regnet', 'res2net', 'resnest']:
        models = get_model_files(os.path.join(cfg_dir, f'configs/{prefix}/metafile.yml'))
        setup(models, '/yueyuxin/mmdetection/corruption_benchmarks', cfg_dir, prefix=prefix)
    """
    for d in os.listdir(root):
        #if d in ['panoptic_fpn', 'ld','lad', 'gn', 'gn+ws', 'seesaw_loss', 'dcn','cascade_rpn','retinanet','faster_rcnn','mask_rcnn']: continue
        p = os.path.join(root, d, 'metafile.yml')
        if not os.path.exists(p): continue
        models = get_model_files(os.path.join(cfg_dir, f'configs/{d}/metafile.yml'))
        setup(models, '/yueyuxin/mmdetection/corruption_benchmarks', cfg_dir, prefix=d)
    """

