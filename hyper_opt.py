'''
Author: Yang Jialong
Date: 2024-11-12 10:15:12
LastEditors: Please set LastEditors
LastEditTime: 2025-02-25 16:22:52
Description: 超参数调优
'''
import os
import json
import torch
import argparse
from torch.utils.data import DataLoader 
from torch import nn
from modules.model import RoadPredictionModel, train_model, test_model, val_model
from dataset.highD.data_processing import HighD, split_dataset, get_label_weight, get_sample_weight, standard_normalization
from dataset.highD.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_DIR
from utils.my_json import save_hyperparams, save_tuning_process
from hyperopt import fmin, tpe, hp,Trials, STATUS_OK


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
    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_ids = [0, 1]
    
    #归一化处理
    scalar = {}
    scalar["target"] = standard_normalization(torch.stack([i['target_obs_traj'] for i in train_set]))
    scalar["surrounding"] = standard_normalization(torch.stack([i['surrounding_obs_traj'] for i in train_set]))

    
    #定义搜索空间
    space = {
        "hidden_size": hp.choice("hidden_size", [32,64,128,256]),
        "style_size": hp.choice("style_size", [8, 16, 32, 64, 128]),
        "decoder_size": hp.choice("decoder_size", [64, 128, 256]),
        "num_layers": hp.choice("num_layers", [1,2,4]),
        "head_num": hp.choice("head_num", [1,2,4]),
        "inputembedding_size": hp.choice("inputembedding_size", [8, 16, 32, 64, 128]), 
        "lr": hp.choice("lr", [0.001]),
        "epoch": hp.choice("epoch", [200]),
        "decay_rate": hp.choice("decay_rate", [0.1, 0.5, 1.0]),
        "decay_step": hp.choice("decay_step", [50, 100]),
        "batch_size": hp.choice("batch_size", [32, 64, 128, 256]),
        "patience": hp.choice("patience", [30]),
    }
    
    #保存路径
    save_path = f'./save/0206_single_modal_hyperopt_horizon={args_input.predict_length}s/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #定义目标函数
    def objective_function(params):
        '''
        description: 目标函数
        return {*}
        '''        
        model = RoadPredictionModel(obs_len=10, 
                                    pred_len=5 * args_input.predict_length,
                                    input_size=10, 
                                    hidden_size=params["hidden_size"], 
                                    num_layers=params["num_layers"], 
                                    head_num=params["head_num"],
                                    inputembedding_size=params["inputembedding_size"],
                                    decoder_size=params["decoder_size"],
                                    style_size=params["style_size"],
                                    predict_trajectory=True,
                                    device=device,
                                    top_k=1,)
        #多卡并行
        # if torch.cuda.device_count() > 1:
        #     print(f"Let's use {torch.cuda.device_count()} GPUs!")
        #     model = nn.DataParallel(model)
        model.to(device)
        train_model(train_set, val_set, model, scalar=scalar, device=device, 
                    save_path=save_path, predict_trajectory=True,  patience=params['patience'], lr=params['lr'], epoch=params['epoch'], 
                    batch_size=params['batch_size'], alpha=1.0, beta=1.0, decay_rate=params['decay_rate'], 
                    decay_step=params['decay_step'])
        #验证集准确率
        model.load_state_dict(torch.load(save_path + "model.pth", map_location=device))
        model.to(device)
        val_ade, val_fde, val_missrate, _, _ = test_model(val_set, model, scalar, predict_trajectory=True, device=device, batch_size = 64, visulization=False)
        
        #实时保存结果
        result = {
            "params": params,
            "ade": val_ade.item(),
            "fde": val_fde.item(),
            "missrate": val_missrate.item()
        }
        save_tuning_process(result, save_path + 'tuning_process.json')
        return {'loss': val_ade, 'params': params, 'status': STATUS_OK}
    
    trials=Trials()
    best = fmin(fn=objective_function, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
    print("最优参数组合为：{}".format(best))
    
    #保存结果
    trails_result = trials.results 
    save_hyperparams(best, save_path=save_path + 'best_hyperparams.json')