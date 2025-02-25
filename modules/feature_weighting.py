'''
Author: Yang Jialong
Date: 2024-11-05 16:34:50
LastEditors: Please set LastEditors
LastEditTime: 2025-02-25 16:34:19
Description: feature attention
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureWeighting(nn.Module):
    def __init__(self, time_step, feature_size, inputembedding_size = 64) -> None:
        super(FeatureWeighting, self).__init__()
        self.embedding = nn.Linear(feature_size, inputembedding_size, dtype=torch.float64)
        self.squeeze = nn.Linear(time_step, 1, dtype=torch.float64)
        self.excitation = nn.Sequential(nn.Linear(inputembedding_size, inputembedding_size // 2, dtype=torch.float64), 
                                        nn.LeakyReLU(0.1),
                                        nn.Linear(inputembedding_size // 2, inputembedding_size, dtype=torch.float64),
                                        nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(inputembedding_size, dtype=torch.float64)
    def forward(self, input: torch.Tensor):
        """
        输入: 时序特征 (bs, t, d)
        """
        input = self.embedding(input)
        input_t = input.permute(0,2,1) #(bs, d, t)       
        squeezed_feature = self.squeeze(input_t).squeeze(-1) #(bs, d)
        attention_score = self.excitation(squeezed_feature).unsqueeze(-1) #(bs, d, 1)
        input_t = input_t * attention_score
        return self.layer_norm(input_t.permute(0,2,1))
    
    
    # def __init__(self, time_step, feature_size, inputembedding_size = 64) -> None:
    #     super(FeatureWeighting, self).__init__()
    #     self.squeeze = nn.Linear(time_step, 1, dtype=torch.float64)
    #     self.excitation = nn.Sequential(
    #                                     nn.Linear(feature_size, feature_size, dtype=torch.float64),
    #                                     nn.Sigmoid())
    #     self.layer_norm = nn.LayerNorm(feature_size, dtype=torch.float64)
    # def forward(self, input: torch.Tensor):
    #     """
    #     输入: 时序特征 (bs, t, d)
    #     """
    #     input_t = input.permute(0,2,1) #(bs, d, t)       
    #     squeezed_feature = self.squeeze(input_t).squeeze(-1) #(bs, d)
    #     attention_score = self.excitation(squeezed_feature).unsqueeze(-1) #(bs, d, 1)
    #     weighted_feature = input_t * attention_score
    #     return self.layer_norm(weighted_feature.permute(0,2,1))
    
        