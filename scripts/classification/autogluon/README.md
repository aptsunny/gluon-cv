## Kaggle Benchmark by GluonCV 

|Datset| Classes/Train/Val/Test | Epochs/Batchsize/GPU |  Model | Gluoncv_Baseline(Score(1st)/Rank & Log & Submission)|
|:-------:|:-----:|:-------:|:-------:|:-------:|
|[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)|2/20000/5000/12500|180/512/4|resnet34_v1b| 0.17131(0.03302)/n% & [log](../autogluon/log_baseline/cats_resnet34_v1b_best.log) & [csv](dogs-vs-cats-redux-kernels-edition/predict.csv)|
|[Aerial Cactus Identification](https://www.kaggle.com/c/aerial-cactus-identification/data)|2/14001/3499/4002 |180/256/4|resnet34_v1b| 0.9711(1.0)/n% & [log](../autogluon/log_baseline/aerial_resnet34_v1b_best.log) & [csv](aerial-cactus-identification/predict.csv)|
|[Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification)|12/3803/947/794 |120/128/2|resnet50_v1| 0.97607(1.0)/n% & [log](../autogluon/log_baseline/plant_resnet50_v1_best.log) & [csv](autogluon_kaggle/autogluon_baseline/plant-seedlings-classification/predict.csv)|
|[The ature Conservancy Fisheries Monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring)|8/3025/752/1000|120/128/2|resnet50_v1|1.01974(0.29535)/n% & [log](../autogluon/log_baseline/fish_resnet50_v1_best.log) & [csv](autogluon_kaggle/autogluon_baseline/fisheries_Monitoring/predict.csv)|
|[Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)|120/8221/2001/10357|180/48/4|resnext101_64x4d| 1.54852(0:extra dataset)/n% & [log](../autogluon/log_baseline/dog_resnext101_64x4d_best.log) & [csv](dog-breed-identification/predict.csv)|
|[Shopee-iet](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/overview) | 18/30567/7636/16111  | 180/48/4|resnet152_v1d| 0.81750(0.87378)/n% & [log](../autogluon/log_baseline/shopee_resnet152_v1d_best.log) & [csv](shopee-iet-machine-learning-competition/predict.csv)|

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

## load_params -> overfit
python train_kaggle_baseline.py --mode hybrid --lr 0.4 --lr-mode cosine --num-gpus 2 -j 60 --warmup-epochs 5 --last-gamma --no-wd --label-smoothing --dtype float32 --classes 18 --num_training_samples 30567 --num-epochs 800 --batch-size 64 --model resnet152_v1d --data-dir /media/ramdisk/dataset/shopee-iet-machine-learning-competition/ --save-dir shopee_params_resnet152_v1d_best_fp32_2 --logging-file shopee_params_resnet152_v1d_best_fp32_2.log --resume-params /home/ubuntu/workspace/baseline_gluoncv_model_saved/shopee_params_resnet152_v1d_best/imagenet-resnet152_v1d-179.params --resume-states /home/ubuntu/workspace/baseline_gluoncv_model_saved/shopee_params_resnet152_v1d_best/imagenet-resnet152_v1d-179.states --resume-epoch 180

## 
```


