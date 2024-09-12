"""
@Time ： 2024/9/12 11:36
@Author ： jtch
"""
# 对yolo数据集进行划分
import os, shutil, random
import numpy as np

TXT_path = 'D:/1-bbox'  # 原TXT文件
Image_path = 'D:/imgs'  # 原图片文件
dataset_path = 'D:/YOLO-BBOX'  # 保存的文件位置
val_size, test_size = 0.1, 0.1

os.makedirs(dataset_path, exist_ok=True)
os.makedirs(f'{dataset_path}/images', exist_ok=True)
os.makedirs(f'{dataset_path}/images/train', exist_ok=True)
os.makedirs(f'{dataset_path}/images/val', exist_ok=True)
os.makedirs(f'{dataset_path}/images/test', exist_ok=True)
os.makedirs(f'{dataset_path}/labels/train', exist_ok=True)
os.makedirs(f'{dataset_path}/labels/val', exist_ok=True)
os.makedirs(f'{dataset_path}/labels/test', exist_ok=True)

path_list = np.array([i.split('.')[0] for i in os.listdir(TXT_path) if 'txt' in i])
random.shuffle(path_list)
train_id = path_list[:int(len(path_list) * (1 - val_size - test_size))]
val_id = path_list[int(len(path_list) * (1 - val_size - test_size)):int(len(path_list) * (1 - test_size))]
test_id = path_list[int(len(path_list) * (1 - test_size)):]

k = 1
print("-------开始划分训练集-------")
for i in train_id:
    print(f"第{k}张图片")
    shutil.copy(f'{Image_path}/{i}.jpg', f'{dataset_path}/images/train/{i}.jpg')
    shutil.copy(f'{TXT_path}/{i}.txt', f'{dataset_path}/labels/train/{i}.txt')
    k += 1

print("-------开始划分验证集-------")
for i in val_id:
    print(f"第{k}张图片")
    shutil.copy(f'{Image_path}/{i}.jpg', f'{dataset_path}/images/val/{i}.jpg')
    shutil.copy(f'{TXT_path}/{i}.txt', f'{dataset_path}/labels/val/{i}.txt')
    k += 1

print("-------开始划分测试集-------")
for i in test_id:
    print(f"第{k}张图片")
    shutil.copy(f'{Image_path}/{i}.jpg', f'{dataset_path}/images/test/{i}.jpg')
    shutil.copy(f'{TXT_path}/{i}.txt', f'{dataset_path}/labels/test/{i}.txt')
    k += 1
