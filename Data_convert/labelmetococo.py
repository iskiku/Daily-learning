"""
@Time ： 2024/9/12 11:29
@Author ： jtch
"""
# labelme标注的多边形 转换为 coco格式
import os
import json
import numpy as np
import glob
import shutil
from sklearn.model_selection import train_test_split

np.random.seed(41)

# 改成自己的类别
# classname_to_id = {
#     'ap-fuselage': 1,
#     'ap-motor': 2,
#     'ap-wing': 3,
#
#     'car-body': 4,
#     'car-cockpit': 5,
#     'bus-body': 6,
#     'bus-cockpit': 7,
#     'tr-body': 8,
#     'tr-head': 9,
#     'ca-board': 10,
#     'ca-deck': 11,
#     'ca-island': 12,
#     'cs-board': 13,
#     'cs-shipbridge': 14,
#     'cs-warehouse': 15,
#     'ws-board': 16,
#     'ws-deck': 17,
#     'ws-shipbridge': 18,
#     'br-deck': 19
# }


classname_to_id = {
    'car-body': 1,
    'car-cockpit': 2,
    'bus-body': 3,
    'bus-cockpit': 4,
    'tr-body': 5,
    'tr-head': 6
}

class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    # 由json文件构建COCO
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                print(json_path)
                annotation = self._annotation(shape)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, obj, path):
        image = {}
        from labelme import utils
        image['height'] = obj['imageHeight']
        image['width'] = obj['imageWidth']
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape):
        label = shape['label']
        # if label not in ['car-body', 'car-cockpit', 'bus-body', 'bus-cockpit', 'tr-body', 'tr-head']:
        #     print(self.image['file_name'])

        points = shape['points']
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        # annotation['category_id'] = self.getcatid(label)
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = annotation['bbox'][-1]*annotation['bbox'][-2]
        return annotation

    # 读取json文件，返回一个json对象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        # print(points)
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


# 训练过程中，如果遇到Index put requires the source and destination dtypes match, got Long for the destination and Int for the source
# 参考：https://github.com/open-mmlab/mmdetection/issues/6706
if __name__ == '__main__':
    labelme_path = "D:/Project-1/datasets/labelme-json-part"
    saved_coco_path = "D:/Project-1/datasets/coco-dj-1"
    print('reading...')

    # 创建文件
    if not os.path.exists("%scoco/annotations/" % saved_coco_path):
        os.makedirs("%scoco/annotations/" % saved_coco_path)
    if not os.path.exists("%scoco/images/train2017/" % saved_coco_path):
        os.makedirs("%scoco/images/train2017" % saved_coco_path)
    if not os.path.exists("%scoco/images/val2017/" % saved_coco_path):
        os.makedirs("%scoco/images/val2017" % saved_coco_path)

    # 获取images目录下所有的joson文件列表
    print(labelme_path + "/*.json")
    json_list_path = glob.glob(labelme_path + "/*.json")
    print('json_list_path: ', len(json_list_path))

    # 数据划分,这里没有区分val2017和tran2017目录，所有图片都放在images目录下
    train_path, val_path = train_test_split(json_list_path, test_size=0.2, train_size=0.8)
    print("train_n:", len(train_path), 'val_n:', len(val_path))

    # 把训练集转化为COCO的json格式
    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2017.json' % saved_coco_path)

    # 把验证集转化为COCO的json格式
    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017.json' % saved_coco_path)

    for file in train_path:
        print("训练图片" + file)
        # img_name = file.replace('json', 'jpg')
        img_name = 'D:\\imgs\\' + file.split('\\')[-1][:-4] + 'jpg'
        print(img_name)
        shutil.copy(img_name, "{}coco/images/train2017/{}".format(saved_coco_path, img_name.split('\\')[-1].replace('png', 'jpg')))

    for file in val_path:
        print("验证图片" + file)
        # img_name = file.replace('json', 'jpg')
        img_name = 'D:\\imgs\\' + file.split('\\')[-1][:-4] + 'jpg'
        print(img_name)
        shutil.copy(img_name, "{}coco/images/val2017/{}".format(saved_coco_path, img_name.split('\\')[-1].replace('png', 'jpg')))

