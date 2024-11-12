'''
Author: Yang Jialong
Date: 2024-11-12 11:13:49
LastEditors: Please set LastEditors
LastEditTime: 2024-11-12 14:55:20
Description: json文件相关
'''

import json
import os

def save_hyperparams(best_params, save_path):
    date = '2024-11-12'
    info = '1'
    result = {
        "date": date,
        "info": info,
        "best_params": best_params
    }
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            results = json.load(f)
    else:
        results = []
    results.append(result)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
        
def save_tuning_process(trails, save_path):
    with open(save_path, "w") as f:
        json.dump(trails, f, indent=4, default=str)