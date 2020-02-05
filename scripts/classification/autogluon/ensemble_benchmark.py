import pandas as pd
from glob import glob
import numpy as np

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='predict a model for different kaggle competitions.')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='training and validation pictures to use.')
    opt = parser.parse_args()
    return opt
opt = parse_args()

def filter_value(value, Threshold):
    if value > Threshold:
        value = value
    else:
        value = 0
    return value

# glob_files = '/home/ubuntu/workspace/dataset/dog-breed-identification/final_standford_dog_300_resnext101_*.csv'
glob_files = '/home/ubuntu/workspace/dataset/dog-breed-identification/final_standford_dog_*.csv'
glob_files = '/home/ubuntu/workspace/dataset/dog-breed-identification/final_standford_dog_256_300_*.csv'
file_list = []
for i, glob_file in enumerate(glob(glob_files)):
    file_list.append(glob_file)


file_list.sort()
for i in range(len(file_list)):
    print(file_list[i])

s1=pd.read_csv(file_list[0],index_col=0)
s2=pd.read_csv(file_list[1],index_col=0)
s3=pd.read_csv(file_list[2],index_col=0)
"""
s4=pd.read_csv(file_list[3],index_col=0)
s5=pd.read_csv(file_list[4],index_col=0) # resnet_152
s6=pd.read_csv(file_list[5],index_col=0) # resnet50_fp32
s7=pd.read_csv(file_list[6],index_col=0)
s8=pd.read_csv(file_list[7],index_col=0)
s9=pd.read_csv(file_list[8],index_col=0)
s10=pd.read_csv(file_list[9],index_col=0)
s11=pd.read_csv(file_list[10],index_col=0)
s12=pd.read_csv(file_list[11],index_col=0)
s13=pd.read_csv(file_list[12],index_col=0)
s14=pd.read_csv(file_list[13],index_col=0)
s15=pd.read_csv(file_list[14],index_col=0)
s16=pd.read_csv(file_list[15],index_col=0)
"""
#

for i in s1.columns.values:
    #s1[i]=s1[i]*0.3 + s2[i]*0.3 + s3[i]*0.4
    # s1[i] = s1[i]* 0.3 + s2[i]*0.1 + s3[i]*0.1 + s5[i]* 0.3 + s6[i]*0.1 + s7[i]*0.1
    #s1[i] = s5[i]* 0.6 + s6[i]*0.2 + s7[i]*0.2
    #s1[i] = s1[i]* 0.2 + s2[i]*0.1 + s3[i]*0.1 + s5[i]* 0.1 + s6[i]*0.1 + s7[i]*0.1 + s9[i]* 0.1 + s10[i]*0.1 + s11[i]*0.1 + s13[i]* 0.1 + s14[i]*0.1 + s15[i]*0.1
    #s1[i] = s1[i]* 0.25 + s5[i]* 0.25 + s9[i]* 0.25 + s13[i]* 0.25
    #s1[i] =  s13[i]
    #s1[i] =  s9[i]* 0.3 + s10[i]*0.1 + s11[i]*0.1 + s13[i]* 0.3 + s14[i]*0.1 + s15[i]*0.1
    s1[i] =  s1[i]* 0.4 + s2[i]*0.3 + s3[i]*0.3
    s1[i] = s1[i].apply(filter_value, Threshold = opt.threshold)

loc_outfile = '/home/ubuntu/workspace/dataset/dog-breed-identification/kaggle_geomean.csv'
# glob_files = glob_files.replace('*','ensem_0')
# print(glob_files)
s1.to_csv(loc_outfile)
print(loc_outfile)
