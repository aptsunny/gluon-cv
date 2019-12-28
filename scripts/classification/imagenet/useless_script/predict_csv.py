import collections
import argparse,os,glob
import math
from mxnet import autograd, gluon, init, nd
import mxnet as mx
from mxnet.gluon import data as gdata, loss as gloss, model_zoo, nn
from gluoncv.model_zoo import get_model
import shutil
import time
import zipfile


transform_test = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(256),
    # 将图像中央的高和宽均为224的正方形区域裁剪出来
    gdata.vision.transforms.CenterCrop(224),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])])
def reorg_dog_data(data_dir,
                   label_file,
                   train_dir,
                   test_dir,
                   input_dir,
                   valid_ratio):
    # 读取训练数据标签
    with open(os.path.join(data_dir, label_file), 'r') as f:
        # 跳过文件头行（栏名称）
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        idx_label = dict(((idx, label) for idx, label in tokens))

    # reorg_train_valid(data_dir, train_dir, input_dir, valid_ratio, idx_label)

    # 整理测试集
    d2l.mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))


"""
--model resnext101_64x4d --saved-params /home/ubuntu/workspace/gluoncv_kaggle_model_paramers/csv/imagenet-resnext101_64x4-179.params 
"""
parser = argparse.ArgumentParser(description='Predict ImageNet classes from a given image')
parser.add_argument('--model', type=str, required=True,
                    help='name of the model to use')
parser.add_argument('--saved-params', type=str, default='',
                    help='path to the saved model parameters')
opt = parser.parse_args()

# load dataset
label_file, train_dir, test_dir = 'labels.csv', 'train', 'test'
data_dir = '/media/ramdisk/dataset/predict_dog'
label_file = os.path.join(data_dir, label_file)
train_dir = os.path.join(data_dir, train_dir)
test_dir = os.path.join(data_dir, test_dir)

input_dir, batch_size, valid_ratio = 'train_valid_test', 128, 0.1

# reorg_dog_data(data_dir, label_file,
#                train_dir, test_dir, input_dir, valid_ratio)

test_ds = gdata.vision.ImageFolderDataset(
    os.path.join(data_dir, 'test'), flag=1)
    # os.path.join(data_dir, input_dir, 'test'), flag=1)

test_iter = gdata.DataLoader(test_ds.transform_first(transform_test),
                             batch_size, shuffle=False, last_batch='keep')

# Load Model
model_name = opt.model
pretrained = True if opt.saved_params == '' else False
classes = 120
net = get_model(model_name, pretrained=pretrained)
if not pretrained:
    with net.name_scope():
        if hasattr(net, 'output'):
            net.output = gluon.nn.Dense(classes)
        else:
            assert hasattr(net, 'fc')
            net.fc = gluon.nn.Dense(classes)
    net.load_parameters(opt.saved_params)
else:
    classes = net.classes

num_gpus = 8
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

preds = []
for data, label in test_iter:
    output_features = net.features(data)
    # output_features = net.features(data.as_in_context(ctx))
    # output = nd.softmax(net.output_new(output_features))
    output = nd.softmax(net.output(output_features))
    preds.extend(output.asnumpy())
# ids = sorted(os.listdir(os.path.join(data_dir, input_dir, 'test/unknown')))
ids = sorted(os.listdir(os.path.join(data_dir, 'test/unknown')))

with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')