'''
Author: Yang Jialong
Date: 2024-11-12 10:15:12
LastEditors: Please set LastEditors
LastEditTime: 2024-11-20 15:03:03
Description: 超参数调优
'''
import os
import torch 
from torch.utils.data import DataLoader 
from modules.model import RoadPredictionModel, train_model, test_model, val_model, cal_acc
from dataset.highD.data_processing import HighD, split_dataset, get_label_weight, get_sample_weight, standard_normalization
from dataset.highD.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.my_json import save_hyperparams, save_tuning_process
from hyperopt import fmin, tpe, hp,Trials, STATUS_OK


if __name__ == '__main__':
    #数据集
    dataset = HighD(raw_data_dir = RAW_DATA_DIR, 
                    processed_dir = PROCESSED_DATA_DIR, 
                    obs_len = 30, 
                    pred_len = 50, 
                    process_data = False)
    train_set, val_set, test_set = split_dataset(dataset)
    
    #device
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
        
        
    #归一化处理
    scalar = {}
    scalar["target"] = standard_normalization(torch.stack([i['target_obs_traj'] for i in train_set]))
    scalar["surrounding"] = standard_normalization(torch.stack([i['surrounding_obs_traj'] for i in train_set]))
    label_weight = get_label_weight(dataset)
    sample_weight = get_sample_weight(train_set, label_weight)
    test_weight = get_sample_weight(test_set, label_weight)
    val_weight = get_sample_weight(val_set, label_weight)
    
    #定义搜索空间
    space = {
        "hidden_size": hp.choice("hidden_size", [16, 32, 64, 128]),
        "num_layers": hp.choice("num_layers", [1, 2, 3]),
        "head_num": hp.choice("head_num", [1, 2, 4, 8]),
        "lr": hp.choice("lr", [0.0001, 0.001, 0.01, 0.1]),
        "epoch": hp.choice("epoch", [50, 100, 150, 200]),
        "batch_size": hp.choice("batch_size", [32, 64, 128, 256, 512]),
        "patience": hp.randint("patience", 10)
    }
    
    save_path = './save/test/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #定义目标函数
    def objective_function(params):
        '''
        description: 目标函数
        return {*}
        '''        
        model = RoadPredictionModel(obs_len=30, 
                                input_size=10, 
                                hidden_size=params["hidden_size"], 
                                num_layers=params["num_layers"], 
                                head_num=params["head_num"], 
                                device=device)
        train_model(train_set, val_set, model, train_sample_weight=sample_weight, scalar=scalar, val_sample_weight=val_weight, 
                    save_path=save_path, device=device, patience=params['patience'], lr=params['lr'], epoch=params['epoch'], batch_size=params['batch_size'])
        #验证集准确率
        val_loss, val_acc = val_model(val_set, model, val_weight, scalar, device)
        return {'loss': -val_acc, 'params': params, 'status': STATUS_OK}
    
    trials=Trials()
    best = fmin(fn=objective_function, space=space, algo=tpe.suggest, max_evals=2, trials=trials)
    print("最优参数组合为：{}".format(best))
    
    #保存结果
    trails_result = trials.results 
    save_tuning_process(trails_result, save_path=save_path + 'tuning_process.json')
    save_hyperparams(best, save_path=save_path + 'best_hyperparams.json')