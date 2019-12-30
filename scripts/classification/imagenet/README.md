# Image Classification on ImageNet

## Inference/Calibration Tutorial

### Float32 Inference

```
python verify_pretrained.py --model=resnet50_v1d_0.11 --batch-size=1
```

### Calibration

Naive calibrate model by using 5 batch data (32 images per batch). Quantized model will be saved into `./model/`.

```
python verify_pretrained.py --model=resnet50_v1d_0.11 --batch-size=32 --calibration
```

### INT8 Inference

```
python verify_pretrained.py --model=resnet50_v1d_0.11 --batch-size=1 --deploy --model-prefix=./model/resnet50_v1d_0.11-quantized-naive
```

## Performance

model | f32 latency(ms) | s8 latency(ms) | f32 throughput(fps, BS=64) | s8 throughput(fps, BS=64) | f32 accuracy | s8 accuracy
-- | -- | -- | -- | -- | -- | --
resnet50_v1 | 11.36 | 2.54 | 190.2 | 1363.75 | 77.21/93.56 | 76.34/93.13
resnet50_v1d_0.11 | 8.84 | 1.74 | 1070.66 | 10686.77 | 63.06/84.64 | 62.68/84.43
mobilenet1.0 | 3.88 | 0.88 | 583.05 | 5615.58 | 73.28/91.22 | 72.23/90.64
mobilenetv2_1.0 | 18.10 | 1.34 | 226.27 | 5005.94 | 71.89/90.53 | 70.87/89.88
squeezenet1.0 | 4.18 | 0.96 | 590.76 | 3393.09 | 57.74/80.33 | 56.98/79.66
squeezenet1.1 | 3.31 | 0.87 | 964.83 | 6027.15 | 58.00/80.47 | 57.02/79.73
inceptionv3 | 20.73 | 4.99 | 156.63 | 917.67 | 78.80/94.37 | 77.36/93.57
vgg16 | 16.71 | 7.63 | 87.17 | 399.62 | 73.06/91.18 | 71.94/90.59

Please refer to [GluonCV Model Zoo](http://gluon-cv.mxnet.io/model_zoo/index.html#image-classification)
for available pretrained models, training hyper-parameters, etc.


## Kaggle Benchmark by GluonCV 

|Datset| Classes/Train/Val/Test | Epochs/Batchsize/GPU |  Model | Gluoncv_Baseline(Score(1st)/Rank & Log)|
|:-------:|:-----:|:-------:|:-------:|:-------:|
|[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)|2/20000/5000/12500|180/512/4|resnet34_v1b| 0.17131(0.03302)/n% & [log](./log_baseline/cats_resnet34_v1b_best.log)|
|[Aerial Cactus Identification](https://www.kaggle.com/c/aerial-cactus-identification/data)|2/14001/3499/4002 |180/256/4|resnet34_v1b| 0.9711(1.0)/n% & [log](./log_baseline/aerial_resnet34_v1b_best.log)|
|[Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification)|12/3803/947/794 |120/128/2|resnet50_v1| 0.97607(1.0)/n% & [log](./log_baseline/plant_resnet50_v1_best.log)|
|[The ature Conservancy Fisheries Monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring)|8/3025/752/1000|120/128/2|resnet50_v1|1.01974(0.29535)/n% & [log](./log_baseline/fish_resnet50_v1_best.log)|
|[Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)|120/8221/2001/10357|180/48/4|resnext101_64x4d| 1.54852(0:extra dataset)/n% & [log](./log_baseline/dog_resnext101_64x4d_best.log)|
|[Shopee-iet](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/overview) | 18/30567/7636/16111  | 180/48/4|resnet152_v1d| 0.81750(0.87378)/n% & [log](./log_baseline/shopee_resnet152_v1d_best.log)|

### Extra Kaggle Dataset Script.
-------------
[Reproduce script](./train_kaggle_baseline.py)

```
## cats
python train_kaggle_baseline.py --use-pretrained --data-dir /media/ramdisk/data/dataset/dogs-vs-cats-redux-kernels-edition/ --model resnet34_v1b --mode hybrid --lr 0.4 --lr-mode step --num-epochs 180 --batch-size 512 --num-gpus 4 -j 60 --warmup-epochs 5 --dtype float32 --last-gamma --no-wd --label-smoothing --save-dir cats_params_resnet34_v1b_best --logging-file cats_resnet34_v1b_best.log

## aerial
python train_kaggle_baseline.py --use-pretrained --data-dir /media/ramdisk/data/dataset/aerial-cactus-identification/ --model resnet34_v1b --mode hybrid --lr 0.4 --lr-mode step --num-epochs 180 --batch-size 256 --num-gpus 4 -j 60 --warmup-epochs 5 --dtype float32 --last-gamma --no-wd --label-smoothing --save-dir aerial_params_resnet34_v1b_best --logging-file aerial_resnet34_v1b_best.log

## plant
python train_kaggle_baseline.py --use-pretrained --data-dir /media/ramdisk/dataset/plant-seedlings-classification/ --model resnet50_v1 --mode hybrid --lr 0.4 --lr-mode cosine --num-epochs 120 --batch-size 128 --num-gpus 2 -j 60 --warmup-epochs 5 --dtype float32 --last-gamma --no-wd --label-smoothing --save-dir plant_params_resnet50_v1_best_2 --logging-file plant_params_resnet50_v1_best_2.log --classes 12 --num_training_samples 3803

## fish
python train_kaggle_baseline.py --use-pretrained --data-dir /media/ramdisk/dataset/fisheries_Monitoring/ --model resnet50_v1 --mode hybrid --lr 0.4 --lr-mode cosine --num-epochs 120 --batch-size 128 --num-gpus 2 -j 60 --warmup-epochs 5 --dtype float32 --last-gamma --no-wd --label-smoothing --save-dir fish_params_resnet50_v1_best_2 --logging-file fish_resnet50_v1_best_2.log --classes 8 --num_training_samples 3025


## dog
python train_kaggle_baseline.py --use-pretrained --data-dir /media/ramdisk/data/dataset/dog-breed-identification/ --model resnext101_64x4d --mode hybrid --lr 0.4 --lr-mode step --num-epochs 180 --batch-size 48 --num-gpus 4 -j 60 --warmup-epochs 5 --dtype float32 --last-gamma --no-wd --label-smoothing --save-dir dog_params_resnext101_64x4d_best --logging-file dog_resnext101_64x4d_best.log

## shopee
python train_kaggle_baseline.py --use-pretrained --data-dir /media/ramdisk/data/dataset/shopee-iet-machine-learning-competition/ --model resnet152_v1d --mode hybrid --lr 0.4 --lr-mode step --num-epochs 180 --batch-size 48 --num-gpus 4 -j 60 --warmup-epochs 5 --dtype float32 --last-gamma --no-wd --label-smoothing --save-dir shopee_params_resnet152_v1d_best --logging-file shopee_resnet152_v1d_best.log
```



