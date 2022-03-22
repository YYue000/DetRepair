import shutil
import os
import copy

from imagecorruptions import get_corruption_names

METHOD = 'finetune_clean+failure' # 'finetune_failure' 
WORKSPACEROOT = '/home/yueyuxin/repair_workspace/mmdetection/repair_benchmarks/'+METHOD

SET2RUN = lambda x:x.replace('/home/yueyuxin','')

MAXEPOCH = 8

if __name__ == '__main__':

    prefix = 'retinanet'
    runsh_str = ''

    for model in ['retinanet_r50_fpn_1x_coco']:
        new_cfg_file = 'retinanet_r50_fpn_1x_coco.py'

        for c in get_corruption_names():
            rd = os.path.join(WORKSPACEROOT, prefix, model, f'{c}-3')

            shutil.copy('test_all2.sh', rd)

            runsh_str += f'cd {SET2RUN(rd)}\n'

            runsh_str+=f'bash train.sh {SET2RUN(new_cfg_file)}\n'
            runsh_str+=f'bash test_all2.sh {SET2RUN(new_cfg_file)} {MAXEPOCH}\n'
            runsh_str += 'python get_mean.py 1 >& sum.log\n'

    with open(os.path.join(WORKSPACEROOT, prefix, 'run_s3.sh'), 'w') as fw:
        fw.write(runsh_str)


