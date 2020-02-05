import os, argparse, shutil
from gluoncv.utils import makedirs
import random
path = '/home/ubuntu/workspace/1107gluoncv_cla/data/dog-breed-identification'
path = '/home/ubuntu/workspace/dataset/dog-breed-identification'
# images -> train/val test
src_path = os.path.join(path, 'images')
train_path = os.path.join(path, 'train')
val_path = os.path.join(path, 'val')
# test_path = os.path.join(path, 'test')
makedirs(train_path)
makedirs(val_path)

def split(full_list,shuffle=False,ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2

fileDir = os.path.join(path, 'images')# 取图片的原始路径
pathDir = os.listdir(fileDir)
for i in pathDir:# classes
    makedirs(os.path.join(train_path, i))
    makedirs(os.path.join(val_path, i))

    targetDir = os.path.join(path, 'images', i)
    imagepathDir = os.listdir(targetDir)
    val_img, train_img = split(imagepathDir, shuffle=True, ratio=0.2)
    for im in train_img:
        im = os.path.join(path, 'images', i, im)
        im_path = im.replace('images/', 'train/').strip('\n')
        shutil.copy(os.path.join(im),
                    os.path.join(im_path))

    for im in val_img:
        im = os.path.join(path, 'images', i, im)
        im_path = im.replace('images/', 'val/').strip('\n')
        shutil.copy(os.path.join(im),
                    os.path.join(im_path))
    print("folder is done",i)
