'''
Author: Yang Jialong
Date: 2024-11-05 16:34:50
LastEditors: 
LastEditTime: 2024-11-11 17:08:16
Description: 请填写简介
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureWeighting(nn.Module):
    def __init__(self, time_step, feature_size) -> None:
        super(FeatureWeighting, self).__init__()
        self.squeeze = nn.Linear(time_step, 1, dtype=torch.float64)
        self.excitation = nn.Sequential(nn.Linear(feature_size, feature_size, dtype=torch.float64), nn.Sigmoid())
    
    def forward(self, input: torch.Tensor):
        """
        输入: 时序特征 (bs, t, d)
        """
        input_t = input.permute(0,2,1) #(bs, d, t)
        squeezed_feature = self.squeeze(input_t).squeeze(-1) #(bs, d)
        attention_score = self.excitation(squeezed_feature).unsqueeze(-1) #(bs, d, 1)
        weighted_feature = input_t * attention_score
        return weighted_feature.permute(0,2,1)
    
        