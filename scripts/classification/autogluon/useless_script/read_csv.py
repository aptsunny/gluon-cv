import pandas as pd
import os

# kaggle = 'dogs-vs-cats-redux-kernels-edition/'
# kaggle = 'aerial-cactus-identification/'
kaggle = 'plant-seedlings-classification/'
# kaggle = 'fisheries_Monitoring/'
# kaggle = 'dog-breed-identification/'
# kaggle = 'shopee-iet-machine-learning-competition/'

# csv_path = os.path.join('/media/ramdisk/dataset/', kaggle,'sample_submission.csv')
csv_path = os.path.join('/media/ramdisk/dataset/', kaggle,'plant-seedlings-classification_predict.csv')
# csv_path = os.path.join('/media/ramdisk/dataset/', kaggle,'aerial-cactus-identification_predict.csv')
# csv_path = os.path.join('/media/ramdisk/dataset/', kaggle,'sample_submission_2.csv')
# csv_path = os.path.join('/media/ramdisk/dataset/', kaggle,'submission_dogs.csv')
# csv_path = os.path.join('/media/ramdisk/dataset/', kaggle,'predict.csv')
# csv_path = os.path.join('/media/ramdisk/dataset/', kaggle,'fisheries_Monitoring_predict.csv')

# csv_path = os.path.join('/media/ramdisk/dataset/predict_dog', 'submission.csv')
# csv_path = os.path.join('/home/ubuntu/workspace/gluoncv_kaggle_model_paramers/csv/test', 'submission.csv')
# csv_path = os.path.join('/home/ubuntu/workspace/train_script/autogluon_1208/examples/image_classification/, kaggle,'predict.csv')

df = pd.read_csv(csv_path)

# row = df[df['id'] == '5ffd97e5cba1d4f7150e21ad01324015'].index.tolist()

print('sss:\n',df.head(20))