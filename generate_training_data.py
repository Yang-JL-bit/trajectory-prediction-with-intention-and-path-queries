'''
Author: Yang Jialong
Date: 2025-01-08 11:20:53
LastEditTime: 2025-01-09 19:15:48
Description: 请填写简介
'''
import os
import pandas as pd 
import argparse
from torch import nn
import torch 
from torch.utils.data import DataLoader 
from modules.model import RoadPredictionModel, train_model, test_model
from dataset.highD.data_processing import HighD, split_dataset, get_label_weight, get_sample_weight, standard_normalization
from dataset.highD.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR

parser = argparse.ArgumentParser(description = " ")
parser.add_argument("--predict_length", type=int, default=3)
args = parser.parse_args()

processed_dir = f"C:\\Users\\Public\\yjl\\master_project\\code\\highD\\processed_data_0109_with_planning_traj_{args.predict_length}s\\"
if not os.path.exists(processed_dir): 
    os.mkdir(processed_dir)

dataset = HighD(raw_data_dir = RAW_DATA_DIR, 
                    processed_dir = processed_dir, 
                    obs_len = 50, 
                    pred_len = 25 * args.predict_length, 
                    process_id=[1,2,3,4,5,6,7,8,13],
                    load_id=[1,2,3,4,5,6,7,8,13],
                    under_sample=10000,
                    traj_sample_rate=5,
                    process_data = True)

