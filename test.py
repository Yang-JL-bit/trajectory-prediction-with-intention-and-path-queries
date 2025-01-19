'''
Author: Yang Jialong
Date: 2024-11-11 17:33:57
LastEditTime: 2025-01-19 11:30:45
Description: 测试模型
'''
import os
import pandas as pd 
from torch import nn
import torch
import random
import argparse
from torch.utils.data import DataLoader 
from modules.model import RoadPredictionModel, test_model, val_model
from dataset.highD.data_processing import HighD, split_dataset, get_label_weight, get_sample_weight, standard_normalization
from dataset.highD.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_DIR
from modules.trajectory_generator import trajectory_generator_by_torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from modules.metrics import cal_minADE, cal_minFDE, cal_miss_rate
torch.set_printoptions(threshold=np.inf)

if __name__ == '__main__': 
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_length', type=int, default=3)
    args_input = parser.parse_args()
    processed_dir = DATASET_DIR + f'processed_data_0102_{args_input.predict_length}s/'
    
    #数据集
    dataset = HighD(raw_data_dir = RAW_DATA_DIR, 
                    processed_dir = processed_dir, 
                    obs_len = 50, 
                    pred_len = 75, 
                    load_id=[1,2,3,4,5,6,7,8,13],
                    under_sample=10000,
                    traj_sample_rate=5,
                    process_data = False)
    train_set, val_set, test_set = split_dataset(dataset)
    print("数据集长度{}, {}, {}".format(len(train_set), len(val_set), len(test_set)))
    
    #归一化处理
    scalar = {}
    scalar["target"] = standard_normalization(torch.stack([i['target_obs_traj'] for i in train_set]))
    scalar["surrounding"] = standard_normalization(torch.stack([i['surrounding_obs_traj'] for i in train_set]))
    
    #device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device_ids = [0, 1]
    
    #是否预测轨迹
    predict_trajectory = True
  
    #加载模型
    top_k = 6
    model = RoadPredictionModel(obs_len=10,
                                pred_len=5 * args_input.predict_length,
                                inputembedding_size=16,
                                input_size=10, 
                                hidden_size=256,
                                num_layers=2, 
                                head_num=1, 
                                style_size=64,
                                decoder_size=256,
                                predict_trajectory=predict_trajectory,
                                refinement_num=1,
                                top_k=top_k,
                                dt = 0.2,
                                device=device)
    
    model.load_state_dict(torch.load(f"./save/0118_training_horizon={args_input.predict_length}s/model.pth", map_location=device))
    model.to(device)
    #模型测试
    # label_weight = get_label_weight(dataset)
    # sample_weight = get_sample_weight(train_set, label_weight)
    # test_weight = get_sample_weight(test_set, label_weight)
    # val_weight = get_sample_weight(val_set, label_weight)
    minADE, minFDE, miss_rate, offroad_rate, offkinametic_rate =  test_model(test_set, model,scalar, 
                                                                            predict_trajectory=predict_trajectory, 
                                                                            batch_size=256,
                                                                            device = device, visulization=False, top_k=top_k)
    print(f"ADE is {minADE}, FDE is {minFDE}, MISS RATE is {miss_rate * 100}%")
    print(f"OFFROAD RATE is {offroad_rate * 100}%")
    print(f"OFFKINAMETIC RATE is {offkinametic_rate * 100}%")
    

