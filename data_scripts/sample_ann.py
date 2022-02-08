import pickle
import json
import numpy as np

def dump_anns(id_list, anns_path):
    anns_tr = {'images':[_ for _ in annotations['images'] if _['id'] in id_list],
        'annotations': [_ for _ in annotations['annotations'] if _['image_id'] in id_list]}
    for _ in ['info', 'licenses', 'categories']:
        anns_tr[_] = annotations[_]
    json.dump(anns_tr, open(anns_path,'w'))



annotations = json.load(open('/yueyuxin/data/coco/annotations/instances_train2017.json'))
all_ids = [_['id'] for _ in annotations['images']]
for SN in [1000, 5000, 10000, 50000]:
    np.random.shuffle(all_ids)
    sample_train = list(all_ids[:SN])
    dump_anns(sample_train, f'/yueyuxin/data/coco/sampled_annotations/instances_train2017_sample{SN}.json')


