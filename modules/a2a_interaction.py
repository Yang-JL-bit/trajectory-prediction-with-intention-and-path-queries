'''
Author: Yang Jialong
Date: 2024-11-11 17:33:54
LastEditors: Please set LastEditors
LastEditTime: 2024-11-12 10:04:03
Description: 请填写简介
'''
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, input_size, head_num, query_key_size, value_size, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.fc_q = nn.Linear(input_size, head_num * query_key_size, dtype=torch.float64)
        self.fc_k = nn.Linear(input_size, head_num * query_key_size, dtype=torch.float64)
        self.fc_v = nn.Linear(input_size, head_num * value_size, dtype=torch.float64)
        self.fc_o = nn.Linear(head_num * query_key_size, input_size, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.input_size = input_size
        self.head_num = head_num
        self.query_key_size = query_key_size
        self.value_size = value_size
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=2)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self,query,key,value,attention_mask=None, attention_weights=None):
        b_s, nq = query.shape[:2]
        nk = key.shape[1]    # nk=nv
        q = self.fc_q(query).view(b_s, nq, self.head_num, self.query_key_size).permute(0, 2, 1, 3)         # (b_s, h, nq, d_k)
        k = self.fc_k(key).view(b_s, nk, self.head_num, self.query_key_size).permute(0, 2, 3, 1)           # (b_s, h, d_k, nk)
        v = self.fc_v(value).view(b_s, nk, self.head_num, self.value_size).permute(0, 2, 1, 3)             # (b_s, h, nk, d_v)
        att = torch.matmul(q, k) / np.sqrt(self.query_key_size)                                            # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.head_num * self.value_size) # (b_s, nq, h*d_v)                                                                                     # (b_s, nq, d_k)
        output = self.fc_o(out)
        return output


class A2A(nn.Module):
    def __init__(self, input_size, hidden_size, head_num) -> None:
        super(A2A,self).__init__()
        self.self_attention = SelfAttention(input_size, head_num, hidden_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size, dtype=torch.float64)
        self.layer_norm = nn.LayerNorm(hidden_size, dtype=torch.float64)
    
    def forward(self, target_feature: torch.Tensor, surrounding_feature):
        """
        target_feature: 目标车辆特征 (bs, feature_size)
        surrounding_feature: 周围车辆的特征 (bs, n_agent, feature_size)
        """
        target_feature = target_feature.unsqueeze(1)
        target_feature = self.self_attention(target_feature, surrounding_feature, surrounding_feature) + target_feature
        return self.layer_norm(target_feature.squeeze(1))

# q = torch.randn([16, 1, 10])
# k = torch.randn([16, 3, 10])
# v = torch.randn([16, 3, 10])
# model = SelfAttention(10, 4, 16, 16)
# output = model(q, k, v)
# print(output.shape)
        