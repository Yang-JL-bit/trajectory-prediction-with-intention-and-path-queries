'''
Author: Yang Jialong
Date: 2024-11-11 17:33:55
LastEditTime: 2025-01-18 16:03:13
Description: 请填写简介
'''
'''
Author: Yang Jialong
Date: 2024-11-11 09:56:15
LastEditors: Please set LastEditors
LastEditTime: 2025-01-12 22:05:48
Description: 训练并测试道路场景下意图识别模型
'''
import os
import pandas as pd 
from torch import nn
import torch 
import argparse
from torch.utils.data import DataLoader 
from modules.model import RoadPredictionModel, train_model, test_model
from dataset.highD.data_processing import HighD, split_dataset, get_label_weight, get_sample_weight, standard_normalization
from dataset.highD.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR,DATASET_DIR


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
                    process_id=[1,2,3,4,5,6,7,8,13],
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
    
    #保存路径
    save_path = f'./save/0118_training_horizon={args_input.predict_length}s/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
           
    #从checkpoint恢复训练
    checkpoint = None
    #checkpoint = torch.load(save_path + 'checkpoint.pth')
    #模型训练
    #多卡并行
    # if torch.cuda.device_count() > 1:
    #     print(f"Let's use {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(model)
    model.to(device)
    # label_weight = get_label_weight(dataset)
    # sample_weight = get_sample_weight(train_set, label_weight)
    # test_weight = get_sample_weight(test_set, label_weight)
    # val_weight = get_sample_weight(val_set, label_weight)
    train_model(train_set, val_set, model, scalar=scalar, 
                save_path=save_path, predict_trajectory=predict_trajectory, device=device, patience=30, batch_size=128, lr=0.001,
                epoch=200, alpha=1.0, beta=1.0, checkpoint=checkpoint, top_k=top_k, decay_step=100, decay_rate=0.1)
    
    
