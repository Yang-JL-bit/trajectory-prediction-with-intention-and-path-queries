'''
Author: Yang Jialong
Date: 2024-11-15 10:22:52
LastEditTime: 2025-02-25 16:31:00
Description: 轨迹生成器，与轨迹微调网络
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from dataset.highD.utils import *
from utils.math import gaussian_elimination, derivative






    
            
class TrajectoryDecoder(nn.Module):
    def __init__(self, input_size, driving_style_hidden_size, hidden_size, num_layers, n_predict, use_traj_prior = False, use_endpoint_prior = False, output_length = 40) -> None:
        super(TrajectoryDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_predict = n_predict
        self.driving_style_hidden_size = driving_style_hidden_size
        self.driving_style = nn.Parameter(torch.randn(1, n_predict, driving_style_hidden_size, dtype=torch.float64))
        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dtype=torch.float64)
        self.traj_head = nn.Linear(hidden_size, 2, dtype=torch.float64)
        self.trajprob_head = nn.Sequential(
                nn.Linear(input_size, self.hidden_size, dtype=torch.float64),
                nn.LeakyReLU(0.01),
                nn.Linear(self.hidden_size, 1, dtype=torch.float64),
            )
        #compare
        # self.traj_head = nn.Linear(hidden_size, 2 * n_predict, dtype=torch.float64)
        # self.trajprob_head = nn.Sequential(
        #     nn.Linear(input_size, self.hidden_size, dtype=torch.float64),
        #     nn.LeakyReLU(0.01),
        #     nn.Linear(self.hidden_size, n_predict, dtype=torch.float64),
        #     )
        #自注意力机制
        self.mode2mode_att = nn.MultiheadAttention(embed_dim=self.driving_style_hidden_size, num_heads=1, batch_first=True, dtype=torch.float64)
        if use_traj_prior or use_endpoint_prior:                
            if use_traj_prior and not use_endpoint_prior:
                self.driving_style_generator = nn.LSTM(input_size=2, 
                                                       hidden_size=self.driving_style_hidden_size, 
                                                       num_layers=1, 
                                                       batch_first=True, 
                                                       dtype=torch.float64)
            elif use_endpoint_prior and not use_traj_prior:
                self.driving_style_generator = nn.Sequential(nn.Linear(2, self.driving_style_hidden_size, dtype=torch.float64))
            else:
                raise ValueError("use_traj_prior and use_endpoint_prior cannot be both True")
    
    '''
    description: 给每条候选轨迹进行解码
    param {*} self
    param {*} obs_feature  车辆的历史观测特征 (bs, hidden_dim)
    return {*} 微调后的轨迹
    # '''
    def forward(self, obs_feature, lane_change_feature, out_length, driving_style_prior = None):
        bs = obs_feature.shape[0]
        if driving_style_prior is None:
            driving_style = self.driving_style.expand(bs, self.n_predict, -1)
        else:
            if driving_style_prior.dim() == 3:
                driving_style = self.driving_style_generator(driving_style_prior)[0][:, -1, :]
                driving_style = driving_style.unsqueeze(0).expand(bs, -1, -1)
            else:
                driving_style = self.driving_style_generator(driving_style_prior)
                driving_style = driving_style.unsqueeze(0).expand(bs, -1, -1)
        driving_style = F.layer_norm(driving_style + self.mode2mode_att(driving_style, driving_style, driving_style)[0], driving_style.size()[-2:])
        obs_feature = obs_feature.unsqueeze(1).expand(-1, self.n_predict, -1)
        lane_change_feature = lane_change_feature.unsqueeze(1).expand(-1, self.n_predict, -1)
        input_ = torch.cat([obs_feature, lane_change_feature, driving_style], dim=-1)
        # input_1, _= self.mode2mode_att(input_, input_, input_)
        # # 残差连接和layer norm
        # input_2 = F.layer_norm(input_ + input_1, input_.size()[-2:])
        lstm_input = input_.contiguous().view(bs * self.n_predict, 1, -1).expand(-1, out_length, -1)
        traj_output, _ = self.lstm_layer(lstm_input)
        traj_output = traj_output.view(bs, self.n_predict, out_length, self.hidden_size)
        traj_output = self.traj_head(traj_output)
        traj_prob_output = self.trajprob_head(input_).squeeze(-1)
        traj_prob_output = F.softmax(traj_prob_output, dim=-1)

        return traj_output, traj_prob_output, traj_output[:, :, -1, :]
    
    # 不适用driving style query
    # def forward(self, obs_feature, lane_change_feature, out_length, driving_style_prior = None):
    #     bs = obs_feature.shape[0]
    #     input_ = torch.cat([obs_feature, lane_change_feature], dim=-1)
    #     lstm_input = input_.contiguous().view(bs, 1, -1).expand(-1, out_length, -1)
    #     traj_output, _ = self.lstm_layer(lstm_input)
    #     traj_output = self.traj_head(traj_output).view(bs, out_length, self.n_predict,2).permute(0,2,1,3)
    #     traj_prob_output = self.trajprob_head(input_)
    #     return traj_output, traj_prob_output, traj_output[:, :, -1, :]


class TrajectoryEvaluator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers) -> None:
        super(TrajectoryEvaluator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size + 3, num_layers=num_layers, batch_first=True, dtype=torch.float64)
        self.score_output_layer = nn.Linear(hidden_size + 3, 1, dtype=torch.float64)
    
    
    '''
    description: 给每条候选轨迹进行打分
    param {*} self
    param {*} obs_feature  车辆的历史观测特征 (bs, hidden_dim)
    param {*} candidate_trajectory  (bs, pred_len, 2)
    param {*} candidate_trajectory_label  (1,3)
    return {*} 打分
    '''
    def forward(self, obs_feature, candidate_trajectory, intention_labal):
        #叠加intention
        intention_labal = intention_labal.repeat(obs_feature.shape[0], 1)
        combined_feature = torch.cat([obs_feature, intention_labal], dim=-1)
        combined_feature = combined_feature.repeat(1, int(candidate_trajectory.shape[0] / obs_feature.shape[0])).reshape(-1, combined_feature.shape[1]) 
        # 使用combined_feature初始化lstm的hidden state, cell_state
        h0 = c0 = combined_feature.unsqueeze(0).repeat(self.num_layers, 1, 1)
        lstm_output, _ = self.lstm_layer(candidate_trajectory, (h0, c0))
        score = self.score_output_layer(lstm_output[:, -1, :])
        return score
    
    
class Time2Centerline(nn.Module):
    def __init__(self, input_dim, driving_style_hidden_size, hidden_dim, n_predict):
        super(Time2Centerline, self).__init__()
        self.input_dim = input_dim
        self.driving_style_hidden_size = driving_style_hidden_size
        self.hidden_dim = hidden_dim
        self.n_predict = n_predict
        self.driving_style = nn.Parameter(torch.randn(1, n_predict, driving_style_hidden_size, dtype=torch.float64))
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=torch.float64),
            nn.LeakyReLU(0.1),
            )
        self.time_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 1, dtype=torch.float64),
            nn.Softplus()
            )
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 1, dtype=torch.float64),
            )
    
    def forward(self, obs_feature, lane_change_feature):
        bs = obs_feature.shape[0]
        driving_style = self.driving_style.repeat(bs, 1, 1)
        obs_feature = obs_feature.unsqueeze(1).repeat(1, self.n_predict, 1)
        lane_change_feature = lane_change_feature.unsqueeze(1).repeat(1, self.n_predict, 1)
        hidden_state = self.hidden_layer(torch.cat([obs_feature, lane_change_feature, driving_style], dim=-1))
        time_output = self.time_predictor(hidden_state).squeeze(-1) + 0.01
        confience_output = self.confidence_predictor(hidden_state).squeeze(-1)
        # confience_output = F.softmax(confience_output, dim=-1)
        return time_output, confience_output
    

