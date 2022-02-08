import pickle
import json
import numpy as np

def dump_anns(id_list, anns_path):
    anns_tr = {'images':[_ for _ in annotations['images'] if _['id'] in id_list],
        'annotations': [_ for _ in annotations['annotations'] if _['image_id'] in id_list]}
    for _ in ['info', 'licenses', 'categories']:
        anns_tr[_] = annotations[_]
    json.dump(anns_tr, open(anns_path,'w'))


SN=1000

annotations = json.load(open('/home/yueyuxin/data/coco/annotations/instances_val2017.json'))
fail_ids = pickle.load(open('/home/yueyuxin/data/coco/c_pickles/retina_fog_5/retina_fog_5_failure_id.pickle','rb')).astype(np.int64)
dump_anns(fail_ids, '/home/yueyuxin/data/coco/c_annotations/retina_fog_5_failure.json')


np.random.shuffle(fail_ids)
repair_train = list(fail_ids[:SN])
dump_anns(repair_train, '/home/yueyuxin/data/coco/c_annotations/retina_fog_5_train.json')
repair_test = list(fail_ids[SN:])
dump_anns(repair_test, '/home/yueyuxin/data/coco/c_annotations/retina_fog_5_test.json')


success_ids = pickle.load(open('/home/yueyuxin/data/coco/c_pickles/retina_fog_5/retina_fog_5_success_id.pickle','rb')).astype(np.int64)
success_ids = list(success_ids)
dump_anns(success_ids, '/home/yueyuxin/data/coco/c_annotations/retina_fog_5_success.json')

dump_anns(success_ids+repair_test, '/home/yueyuxin/data/coco/c_annotations/retina_fog_5_test_mix.json')

