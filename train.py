'''
Author: Yang Jialong
Date: 2024-11-11 09:56:15
LastEditors: Please set LastEditors
LastEditTime: 2024-11-12 16:12:48
Description: 训练并测试道路场景下意图识别模型
'''
import os
import torch 
from torch.utils.data import DataLoader 
from modules.model import RoadPredictionModel, train_model, test_model
from dataset.highD.data_processing import HighD, split_dataset, get_label_weight, get_sample_weight, standard_normalization
from dataset.highD.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR


if __name__ == '__main__':
    #数据集
    dataset = HighD(raw_data_dir = RAW_DATA_DIR, 
                    processed_dir = PROCESSED_DATA_DIR, 
                    obs_len = 50, 
                    pred_len = 75, 
                    process_data = True)
    train_set, val_set, test_set = split_dataset(dataset)
    print("数据集长度{}, {}, {}".format(len(train_set), len(val_set), len(test_set)))
    
    #归一化处理
    scalar = {}
    scalar["target"] = standard_normalization(torch.stack([i['target_obs_traj'] for i in train_set]))
    scalar["surrounding"] = standard_normalization(torch.stack([i['surrounding_obs_traj'] for i in train_set]))
    
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    
    model = RoadPredictionModel(obs_len=30, 
                                input_size=10, 
                                hidden_size=32, 
                                num_layers=2, 
                                head_num=4, 
                                device=device)
    
    save_path = './save/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    #模型训练
    label_weight = get_label_weight(dataset)
    sample_weight = get_sample_weight(train_set, label_weight)
    test_weight = get_sample_weight(test_set, label_weight)
    val_weight = get_sample_weight(val_set, label_weight)
    train_model(train_set, val_set, model, train_sample_weight=sample_weight, scalar=scalar, val_sample_weight=val_weight, save_path=save_path, device=device, patience=7)
    
    #模型测试
    # model = torch.load(save_path + 'model.pth')
    test_accuracy = test_model(test_set, model, test_weight,scalar,  device)
    print("测试准确率为: {}%".format(test_accuracy * 100))