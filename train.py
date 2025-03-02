
'''
Author: Yang Jialong
Date: 2024-11-11 09:56:15
LastEditors: Please set LastEditors
LastEditTime: 2025-02-25 16:25:41
Description: 训练轨迹预测模型
'''
import os
import pandas as pd 
from torch import nn
import torch 
import argparse
import random
from torch.utils.data import DataLoader, Subset
from modules.model import RoadPredictionModel, train_model
from dataset.highD.data_processing import HighD, split_dataset, standard_normalization
from dataset.highD.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_DIR
from modules.traj_cluster import cluster_trajectories, cluster_last_points_by_label



if __name__ == '__main__': 
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_length', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=6)
    parser.add_argument('--inputembedding_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--style_size', type=int, default=64)
    parser.add_argument('--head_num', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--decoder_size', type=int, default=256)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_step', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--predict_trajectory', type=bool, default=True)
    parser.add_argument('--use_driving_style_prior', type=bool, default=True)
    parser.add_argument('--use_endpoint_prior', type=bool, default=False)
    parser.add_argument('--use_trajectory_prior', type=bool, default=True)
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                
    
    #是否预测轨迹
    predict_trajectory = args_input.predict_trajectory
    top_k = args_input.top_k  #top k trajectories


    #轨迹聚类
    if args_input.use_driving_style_prior:
        sample_size = 5000
        random.seed(100)
        indices = random.sample(range(len(train_set)), sample_size)  # 随机选择索引
        sub_trainset = Subset(train_set, indices)  # 创建子集
        if args_input.use_trajectory_prior:
            driving_style_prior = cluster_trajectories(sub_trainset, num_clusters=top_k)
        else:
            driving_style_prior = cluster_last_points_by_label(sub_trainset, num_clusters=top_k)
        print("finish clustering!")
    else:
        driving_style_prior = None

    
    model = RoadPredictionModel(obs_len=10,
                                pred_len=5 * args_input.predict_length,
                                inputembedding_size=args_input.inputembedding_size,
                                input_size=10, 
                                hidden_size=args_input.hidden_size,
                                num_layers=1, 
                                head_num=args_input.head_num, 
                                style_size=args_input.style_size,
                                decoder_size=args_input.decoder_size,
                                predict_trajectory=predict_trajectory,
                                refinement_num=1,
                                top_k=top_k,
                                dt = 0.2,
                                device=device,
                                use_traj_prior=args_input.use_trajectory_prior,
                                use_endpoint_prior=args_input.use_endpoint_prior)
    
    
    #保存路径
    save_path = f'./save/0225_training_horizon={args_input.predict_length}s_baseline_test/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
           
    #从checkpoint恢复训练(None即为重新训练)
    checkpoint = None
    # checkpoint = torch.load(save_path + 'checkpoint.pth')
    
    #模型训练
    #多卡并行
    # if torch.cuda.device_count() > 1:
    #     print(f"Let's use {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(model)
    model.to(device)
    train_model(train_set, 
                val_set, 
                model, 
                scalar=scalar, 
                save_path=save_path, 
                predict_trajectory=predict_trajectory, 
                driving_style_prior=driving_style_prior, 
                device=device, 
                patience=args_input.patience, 
                batch_size=args_input.batch_size, 
                lr=args_input.lr,
                epoch=args_input.epoch, 
                checkpoint=checkpoint, 
                top_k=top_k, 
                decay_step=args_input.decay_step, 
                decay_rate=args_input.decay_rate)
    
    
