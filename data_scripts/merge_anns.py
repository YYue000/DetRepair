import json

f1 = '/yueyuxin/data/coco/sampled_annotations/instances_train2017_sample1000.json'
f2 = '/yueyuxin/data/coco/c_annotations/retina_fog_5_1k_clean_ref_1k.json'
anns_path='/yueyuxin/data/coco/c_annotations/retina_fog_5_1k_clean_ref_1k_train_1k.json'

def get_data(f, img_root=None):
    data = json.load(open(f))
    if img_root is not None:
        for img in data['images']:
            img['file_name'] = img_root+img['file_name']
    return data

data1 = get_data(f1, img_root='train2017/')
data2 = get_data(f2)
anns_tr = {'images': data1['images']+data2['images'] ,
    'annotations': data1['annotations']+data2['annotations']}
for _ in ['info', 'licenses', 'categories']:
    anns_tr[_] = data1[_]
json.dump(anns_tr, open(anns_path,'w'))


