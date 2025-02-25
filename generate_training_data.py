'''
Author: Yang Jialong
Date: 2025-01-08 11:20:53
LastEditTime: 2025-02-25 16:24:51
Description: 生成数据集（从大的数据集中随机抽取一部分进行实验）
'''
import argparse
from dataset.highD.data_processing import HighD
from dataset.highD.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR,DATASET_DIR
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--predict_length', type=int, default=8)
args_input = parser.parse_args()

#数据的保存路径
processed_dir = DATASET_DIR + f'processed_data_0102_{args_input.predict_length}s/'


dataset = HighD(raw_data_dir = RAW_DATA_DIR, 
                    processed_dir = processed_dir, 
                    obs_len = 50, 
                    pred_len = 25 * args_input.predict_length, 
                    process_id=[1,2,3,4,5,6,7,8,13],  #加载文件id
                    load_id=[1, 2, 13],
                    under_sample=10000,  #下采样样本数
                    traj_sample_rate=5,
                    process_data = True
                    )

