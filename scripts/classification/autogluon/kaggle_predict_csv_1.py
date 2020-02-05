import argparse, os, glob, csv
import pandas as pd
import mxnet as mx
import d2lzh as d2l
from mxnet import nd, image, gluon
from mxnet.gluon import data as gdata
from gluoncv.model_zoo import get_model
import random
"""
--custom
pretrain_standford_dog_180_resnext101
--dataset
dog-breed-identification
--model
resnext101_64x4d
--saved-params
/home/ubuntu/workspace/gluon-cv_args/scripts/classification/autogluon/pretrain_dog_params_resnext101_64x4d_best_stanford_fp32/imagenet-resnext101_64x4d-179.params
--classes
120
--dtype
float32
--data-dir
/home/ubuntu/workspace/dataset

# 在不同机器上还要配置数据集位置
python kaggle_predict_csv.py --custom pretrain_standford_dog_180_resnext101 --dataset dog-breed-identification --model resnext101_64x4d --saved-params /home/ubuntu/workspace/gluon-cv_args/scripts/classification/autogluon/pretrain_dog_params_resnext101_64x4d_best_stanford_fp32/imagenet-resnext101_64x4d-179.params --classes 120 --dtype float32 --data-dir /home/ubuntu/workspace/dataset

--random --resize 224 --resize-ratio 0.875 

# [3]
# update 八卡机器的命令
python kaggle_predict_csv.py --custom final_standford_dog_300_resnext101 --dataset dog-breed-identification --model resnext101_64x4d --saved-params /home/ubuntu/workspace/gluon-cv_args/scripts/classification/autogluon/final_fit_model/final_dog_params_resnext101_64x4d_best_fp32/imagenet-resnext101_64x4d-299.params --classes 120 --dtype float32 --data-dir /home/ubuntu/workspace/dataset --resize 224 --resize-ratio 0.875 --random

# [2]ok
python kaggle_predict_csv_old.py --custom final_standford_dog_300_resnext101 --dataset dog-breed-identification --model resnext101_64x4d --saved-params /home/ubuntu/workspace/gluon-cv_args/scripts/classification/autogluon/final_fit_model/final_dog_params_resnext101_64x4d_best_fp32/imagenet-resnext101_64x4d-299.params --classes 120 --dtype float32 --data-dir /home/ubuntu/workspace/dataset

# [1]
# dog ok 
python kaggle_predict_csv.py --custom baseline_dog_180_resnext101 --dataset dog-breed-identification --model resnext101_64x4d --saved-params /home/ubuntu/workspace/baseline_gluoncv_model_saved/dog_params_resnext101_64x4d_best/imagenet-resnext101_64x4d-179.params --classes 120 --dtype float32 --data-dir /home/ubuntu/workspace/dataset --resize 224 --resize-ratio 0.875 --random 


# fish 
python kaggle_predict_csv.py --custom fish_params_resnet50_v1_best_2_119 --dataset fisheries_Monitoring --model resnet50_v1 --saved-params /home/ubuntu/workspace/baseline_gluoncv_model_saved/fish_params_resnet50_v1_best_2_gpu/imagenet-resnet50_v1-119.params --classes 8 --dtype float32 --data-dir /home/ubuntu/workspace/dataset --resize 320

# shopee_params_resnet152_v1d_best
python kaggle_predict_csv.py --custom shopee_params_resnet152_v1d_best_179 --dataset shopee-iet-machine-learning-competition --model resnet152_v1d --saved-params /home/ubuntu/workspace/baseline_gluoncv_model_saved/shopee_params_resnet152_v1d_best/imagenet-resnet152_v1d-179.params --classes 18 --dtype float32 --data-dir /home/ubuntu/workspace/dataset

# aerial_params_resnet34_v1b_best  179 169 159
python kaggle_predict_csv.py --custom aerial_params_resnet34_v1b_best_159 --dataset aerial-cactus-identification --model resnet34_v1b --saved-params /home/ubuntu/workspace/baseline_gluoncv_model_saved/aerial_params_resnet34_v1b_best/imagenet-resnet34_v1b-159.params --classes 2 --dtype float32 --data-dir /home/ubuntu/workspace/dataset

# dogs-vs-cats-redux-kernels-edition 169 179 159
python kaggle_predict_csv.py --custom cats_params_resnet34_v1b_best_159 --dataset dogs-vs-cats-redux-kernels-edition --model resnet34_v1b --saved-params /home/ubuntu/workspace/baseline_gluoncv_model_saved/cats_params_resnet34_v1b_best/imagenet-resnet34_v1b-159.params --classes 2 --dtype float32 --data-dir /home/ubuntu/workspace/dataset

# plant 119 109 99
python kaggle_predict_csv.py --custom plant_fp32_ep120_baseline_gpu_2_99 --dataset plant-seedlings-classification --model resnet50_v1 --saved-params /home/ubuntu/workspace/baseline_gluoncv_model_saved/plant_params_resnet50_v1_best_2_gpu/imagenet-resnet50_v1-99.params --classes 12 --dtype float32 --data-dir /home/ubuntu/workspace/dataset


"""

def parse_args():
    parser = argparse.ArgumentParser(description='Predict kaggle and predict to generate csv')
    parser.add_argument('--model', type=str, required=True,
                        help='name of the model to use')
    parser.add_argument('--saved-params', type=str, default='',
                        help='path to the saved model parameters')
    parser.add_argument('--data-dir', type=str, default= '/media/ramdisk/dataset/',
                        help='path to the input picture')
    parser.add_argument('--dataset', type=str, default = 'shopee-iet-machine-learning-competition',
                        help='path to the input picture')
    parser.add_argument('--dtype', type=str, default = 'float32',
                        help='path to the input picture')
    parser.add_argument('--saved-dir', type=str, default='/home/ubuntu/workspace/baseline_gluoncv_model_saved',
                        help='path to the saved model parameters')
    parser.add_argument('--classes', type=int, default = 18,
                        help='path to the input picture')
    parser.add_argument('--batch-size', type=int, default = 64,
                        help='path to the input picture')
    parser.add_argument('--custom', type=str, default = 'predict',
                        help='path to the input picture')

    parser.add_argument('--resize-ratio', type=float, default = 0.875,
                        help='path to the input picture')
    parser.add_argument('--resize', type=int, default = 224,
                        help='path to the input picture')
    parser.add_argument('--random_crop', action='store_true',
                        help='enable using pretrained model from gluon.')


    opt = parser.parse_args()
    return opt

def load_model(opt, pretrained, ctx):
    # get_model
    net = get_model(opt.model, pretrained=pretrained, ctx=ctx)

    with net.name_scope():
        if hasattr(net, 'output'):
            net.output = gluon.nn.Dense(opt.classes)
        else:
            assert hasattr(net, 'fc')
            net.fc = gluon.nn.Dense(opt.clasd10ses)
    # fp16-> 1
    net.cast(opt.dtype)# >?
    saved_params = os.path.join(opt.saved_dir, opt.saved_params)
    net.load_parameters(saved_params, ctx=ctx)

    return net

def data_iter(opt):
    # data_loader
    test_path = os.path.join(opt.data_dir, opt.dataset, 'test')
    train_path = os.path.join(opt.data_dir, opt.dataset, 'train')
    train_ds = gdata.vision.ImageFolderDataset(train_path, flag=1)
    test_ds = gdata.vision.ImageFolderDataset(test_path, flag=1)

    if opt.random_crop:
        transform_test = gdata.vision.transforms.Compose([
            gdata.vision.transforms.Resize( int(opt.resize / opt.resize_ratio)),  #
            gdata.vision.transforms.RandomResizedCrop(opt.resize),
            gdata.vision.transforms.ToTensor(),
            gdata.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])])
    else:
        transform_test = gdata.vision.transforms.Compose([
            # resize 256/320 再进行inference 或许会好一点？ /0.875
            # gdata.vision.transforms.Resize(256),
            # gdata.vision.transforms.Resize(320),# better 是因为训练resize 480 crop224，所以320 crop 224好一点
            # 将图像中央的高和宽均为224的正方形区域裁剪出来
            # gdata.vision.transforms.RandomResizedCrop(224),  # ensemble
            gdata.vision.transforms.Resize(int(opt.resize / opt.resize_ratio)),  #
            gdata.vision.transforms.CenterCrop(opt.resize),
            # transforms.CenterCrop(input_size),
            gdata.vision.transforms.ToTensor(),
            gdata.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])])
    # mxnet/gluon/data/dataloader.py
    test_iter = gdata.DataLoader(test_ds.transform_first(transform_test), opt.batch_size, shuffle=False,
                                 last_batch='keep')  # shuffle = False
    return test_iter, train_ds

def get_result(train_ds, test_iter, net, ctx, dtype):
    nums_test_iter = len(test_iter)
    times = 0
    preds = []
    pred_cla = []
    inds = []
    value = []
    for data, label in test_iter:
        print("process:", times / nums_test_iter)
        # (128,3,224,224)->(128,18)
        # fp16-> 2
        data = data.astype(dtype, copy=False)

        output_features = net(data.as_in_context(ctx))
        output = nd.softmax(output_features)

        # (128,18)->(128,1)
        ind = nd.argmax(output, axis=1).astype('int')
        # 每个batch_id 对应class
        idx = mx.nd.stack(mx.nd.arange(output.shape[0], ctx=output.context), ind.astype('float32'))
        # 每个batch_id 对应class的概率
        probai = mx.nd.gather_nd(output, idx)

        preds.extend(output.asnumpy())
        pred_cla.extend(probai.asnumpy())
        inds.extend(ind.asnumpy())
        times = times + 1

    # 所有类别的概率
    preds = preds
    # top1 的概率
    pred_cla = pred_cla
    # top1 id
    inds = inds
    # top1 的class_name
    for i in inds:
        value.append(train_ds.synsets[i])
    value = value
    return preds, pred_cla, inds, value

def recorrect_classs_name(csv_path, start, stop, fullname=False):
    df = pd.read_csv(csv_path)
    df['id'] = df['id'].apply(lambda x: int(x[start:stop]))
    df.sort_values("id", inplace=True)
    # if fullname:
    #    df['id'] = df['id'].apply(lambda x: str(x)+'.jpg')

    df.to_csv(csv_path, index=False)

def generate_csv(dataset, csv_path, ids, inds, preds, pred_cla, class_name, custom, resize, random_crop, resize_ratio):
    ## 配置 不同的csv格式
    if dataset == 'dogs-vs-cats-redux-kernels-edition':
        csv_config = {'fullname': False,# -> recorrect_classs_name
                      'need_sample': False,
                      'image_column_name': 'id',
                      'content': 'int',
                      'class_column_name': 'label',
                      'value': 'probability_1',
                      'special': 0 # 读取文件名第一个字符开始
                      }

    elif dataset == 'aerial-cactus-identification':
        csv_config = {'fullname': True,
                      'need_sample': False,
                      'image_column_name': 'id',
                      'content': 'empty',
                      'class_column_name': 'has_cactus',
                      'value': 'probability_1',
                      'special': 0 # 读取文件名第一个字符开始
                      }

    elif dataset == 'plant-seedlings-classification':
        csv_config = {'fullname': True,
                      'need_sample': True,
                      'image_column_name': 'file',
                      'content': 'empty',
                      'class_column_name': 'species',
                      'value': 'category'
                      }
    elif dataset == 'fisheries_Monitoring':
        csv_config = {'fullname': True,
                      'need_sample': True,
                      'image_column_name': 'image',
                      'content': 'str',
                      'class_column_name': '',
                      'value': 'multi_prob'
                      }
    elif dataset == 'dog-breed-identification':
        csv_config = {'fullname': False,
                      'need_sample': True,
                      'image_column_name': 'id',
                      'class_column_name': '',
                      'content': 'str',  # empty
                      'value': 'multi_prob'
                      }
    elif dataset == 'shopee-iet-machine-learning-competition':
        csv_config = {'fullname': False,
                      'image_column_name': 'id',
                      'class_column_name': 'category',
                      'need_sample': False,
                      'content': 'special',
                      'value': 'class_id',
                      'special': 5  # 读取文件名第一个字符开始
                      }
    # 是否需要原来sample submission的信息（标题的类别顺序可能实际读取的不一致，或者是要多类的概率）
    if random_crop:
        # index = random.sample(range(1, 10), 1)
        index = random.randint(1,10)
    else:
        index = 1
    save_csv_name = custom + '_' + str(resize) + '_' + str(index) + '_' + str(resize_ratio)[2:] + '.csv'

    print(csv_path.replace('sample_submission.csv', save_csv_name))
    if csv_config['need_sample']:
        df = pd.read_csv(csv_path)
        # 是否存放在csv里边的是全名 ids->test 文件夹内部按照读取文件名排序
        if not csv_config['fullname']:
            imagename_list = [name_id[:-4] for name_id in ids]
        else:
            imagename_list = ids

        # 读取文件名排序对应的csv的index_list
        row_index_group = []
        for i in imagename_list:
            if csv_config['content'] == 'str':
                row_index = df[df[csv_config['image_column_name']] == str(i)].index.tolist()
            elif csv_config['content'] == 'empty':
                row_index = df[df[csv_config['image_column_name']] == i].index.tolist()
            elif csv_config['content'] == 'int':
                row_index = df[df[csv_config['image_column_name']] == int(i)].index.tolist()
            elif csv_config['content'] == 'special':
                row_index = df[df[csv_config['image_column_name']] == int(i[5:])].index.tolist()
            row_index_group.append(row_index[0]) # 假如读取id在csv只有一个对应行

        # value
        if csv_config['value'] == 'category':
            df.loc[row_index_group, csv_config['class_column_name']] = class_name

        elif csv_config['value'] == 'multi_prob':
            df.loc[row_index_group, 1:] = preds

        if dataset == 'fisheries_Monitoring':
            def get_name(name):
                if name.startswith('image'):
                    name = 'test_stg2/' + name
                return name
            df['image'] = df['image'].apply(get_name)

        df.to_csv(csv_path.replace('sample_submission.csv', save_csv_name), index=False)
        print('predict.csv is done')

    else:
        csv_path = csv_path.replace('sample_submission.csv', save_csv_name)
        with open(csv_path, 'w') as f:
            row = [csv_config['image_column_name'], csv_config['class_column_name']]
            writer = csv.writer(f)
            writer.writerow(row)
            if csv_config['value'] == 'class_id':
                for i, prob in zip(ids, inds):
                    row = [i, prob]
                    writer = csv.writer(f)
                    writer.writerow(row)
            if csv_config['value'] == 'probability_1':
                for i, prob in zip(ids, preds):
                    row = [i, prob[1]]  ###
                    writer = csv.writer(f)
                    writer.writerow(row)
            if csv_config['value'] == 'probability_0':
                for i, prob in zip(ids, preds):
                    row = [i, prob[0]]  ###### 提交方式有问题 kernal
                    writer = csv.writer(f)
                    writer.writerow(row)
        f.close()
        ## 把实际图像读入的名字纠正过来
        if not csv_config['fullname']:
            recorrect_classs_name(csv_path, csv_config['special'], -4)

def main():
    # CLI
    opt = parse_args()
    # Load Model & 或许检查一下预训练模型
    pretrained = True if opt.saved_params == '' else False

    ## ?
    ctx = d2l.try_gpu()  #
    # ctx = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3), mx.gpu(4), mx.gpu(5), mx.gpu(6), mx.gpu(7)]
    # ctx = [mx.gpu(6), mx.gpu(7)]

    ## load trained model
    net = load_model(opt, pretrained, ctx)

    ## test_data-> iteration
    target_test_iter, train_ds = data_iter(opt)

    ## get_result & 准备输出的各项数据
    test_path = os.path.join(opt.data_dir, opt.dataset, 'test')
    ids = sorted(os.listdir(os.path.join(test_path,'no_class')))
    # print(ids)

    preds, pred_cla, inds, value = get_result(train_ds, target_test_iter, net, ctx = d2l.try_gpu(), dtype = opt.dtype)

    ## generate_csv, kaggle要求比较多
    csv_path = os.path.join(opt.data_dir, opt.dataset, 'sample_submission.csv')
    generate_csv(opt.dataset, csv_path, ids, inds, preds, pred_cla, value, opt.custom, opt.resize, opt.random_crop, opt.resize_ratio)

if __name__ == '__main__':
    main()

# 给的命名
# df = pd.DataFrame({'id': sorted_ids, 'label': preds})
# df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
# df.to_csv('submission.csv', index=False)

# 有顺序问题的版本
# def generate_csv(inds, path):
#     with open(path, 'w') as f:
#         row = ['id', 'category']
#         writer = csv.writer(f)
#         writer.writerow(row)
#         id = 1
#         for ind in inds:
#             # row = [id, ind.asscalar()]
#             row = [id, ind]
#             writer = csv.writer(f)
#             writer.writerow(row)
#             id += 1
#     f.close()
# csv_path = os.path.join(opt.input_pic, 'submission.csv')
# generate_csv(inds, csv_path)



