'''
Author: Yang Jialong
Date: 2024-11-15 10:22:52
LastEditTime: 2025-01-11 21:10:02
Description: 轨迹生成器，与轨迹微调网络
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from dataset.highD.utils import *
from utils.math import gaussian_elimination, derivative

class TrajectoryGenerator:
    
    def __init__(self, obs_len, pred_len, lane_info) -> None:
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.lane_info = lane_info
        self.k_max = 1 / 3
    
    '''
    description: 根据当前状态以及变道意图生成未来轨迹集合
    param {*} self
    param {*} target_track_csv:  目标车辆的轨迹(dict)
    param {*} start_frame_idx: 起始帧
    param {*} target_lane_id: 目标车辆未来的目标车道
    return {*}
    '''
    def generate_future_trajectory(self, target_track_csv, start_frame_idx, target_lane_id, downsample_rate):
        predict_frame_idx = start_frame_idx + self.obs_len - 1
        while ((predict_frame_idx - start_frame_idx) % downsample_rate != 0):
            predict_frame_idx = predict_frame_idx - 1
        # 1. 获取轨迹的起始状态
        start_state = {}
        start_state['x'] = target_track_csv[X][predict_frame_idx] + target_track_csv[WIDTH][predict_frame_idx] / 2
        start_state['vx'] = target_track_csv[X_VELOCITY][predict_frame_idx]
        start_state['ax'] = target_track_csv[X_ACCELERATION][predict_frame_idx]
        start_state['y'] = target_track_csv[Y][predict_frame_idx] + target_track_csv[HEIGHT][predict_frame_idx] / 2
        start_state['vy'] = target_track_csv[Y_VELOCITY][predict_frame_idx]
        start_state['ay'] = target_track_csv[Y_ACCELERATION][predict_frame_idx]
        
        # 2. 获取轨迹的变道采样时间
        candidate_time_list = self.get_candidate_time()
        # 3. 获取未来变道轨迹集合
        future_trajectory_list = []
        for candidate_time in candidate_time_list:
            # 3. 获取轨迹的终止状态
            end_state = {}
            end_state['vx'] = start_state['vx'] + candidate_time * start_state['ax']
            end_state['ax'] = start_state['ax']
            end_state['y'] = self.lane_info[target_lane_id][1]
            end_state['vy'] = 0
            end_state['ay'] = 0
            # 4. 分别获取横向和纵向轨迹多项式系数
            lon_coef = self.get_lon_traj_coef(start_state, end_state, candidate_time)
            lat_coef = self.get_lat_traj_coef(start_state, end_state, candidate_time)
            # 5. 检查轨迹是否满足运动学约束， 满足的情况下采样轨迹
            if (True):
                trajectory_points = self.sample_trajectory_from_coef(lon_coef, lat_coef, candidate_time, downsample_rate)               
                future_trajectory_list.append(trajectory_points)
        # return future_trajectory_list
        return self.nms_on_trajectories(future_trajectory_list, 0.2)
        
            
        
    '''
    description: 获取纵向四次多项式系数
    param {*} self
    param {*} start_state 轨迹初始条件
    param {*} end_state 轨迹终止条件
    param {*} t 变道时间
    return {*} coef 系数字典
    '''
    def get_lon_traj_coef(self, start_state, end_state, t):
        coef = {}
        coef[0] = start_state['x']
        coef[1] = start_state['vx']
        coef[2] = start_state['ax'] / 2
        b1 = end_state['vx'] - coef[1] - 2 * t * coef[2]
        b2 = end_state['ax'] - 2 * coef[2]
        coef[3]  = (3 * b1 - b2 * t) / (3 * t * t)
        coef[4] = (b2 * t - 2 * b1) / (4 * t * t * t)
        return coef
    
    
    '''
    description: 获取横向五次多项式系数
    param {*} self
    param {*} start_state 初始状态
    param {*} end_state 终止状态
    param {*} t 变道时间
    return {*} 系数字典
    '''
    def get_lat_traj_coef(self, start_state, end_state, t):
        coef = {}
        coef[0] = start_state['y']
        coef[1] = start_state['vy']
        coef[2] = start_state['ay'] / 2
        b1 = end_state['y'] - coef[0] - coef[1] * t - coef[2] * t * t
        b2 = end_state['vy'] - coef[1] - 2 * coef[2] * t
        b3 = end_state['ay'] - 2 * coef[2]
        #使用高斯消元求剩下的几个变量 Ax = b
        a = np.array([[t ** 5, t ** 4, t ** 3],
                     [5 * (t ** 4), 4 * (t ** 3), 3 * (t ** 2)],
                     [20 * (t ** 3), 12 * (t ** 2), 6 * t]])
        b = np.array([b1, b2, b3])
        x = gaussian_elimination(a, b)
        coef[3] = x[2]
        coef[4] = x[1]
        coef[5] = x[0]
        return coef
    
    '''
    description: 获取所有的变道采样时间
    return {*} candidate_time: 所有的变道采样时间
    '''
    def get_candidate_time(self):
        candidate_time = []
        t = MIN_LANE_CHANGE_TIME
        while (t <= MAX_LANE_CHANGE_TIME):
            candidate_time.append(t)
            t += LANE_CHANGE_SAMPLR_STEP
        return candidate_time
    
    '''
    description: 根据多项式离散采样出轨迹
    param {*} self
    param {*} lon_coef 纵向轨迹多项式系数
    param {*} lat_coef 横向轨迹多项式系数
    param {*} t_max 最大时间
    return {*} 预测出的相对当前位置的轨迹坐标序列
    '''
    def sample_trajectory_from_coef(self, lon_coef, lat_coef, t_max, downsample_rate):
        trajectory_points = []
        cur_lon = lon_coef[0]
        cur_lat = lat_coef[0]
        end_lon_v = lon_coef[1] + t_max * lon_coef[2] * 2
        driving_direction = 1 if lon_coef[1] < 0 else 2
        t = 1 / SAMPLE_RATE * downsample_rate
        step = 1 / SAMPLE_RATE
        i = 0
        while (i < self.pred_len):
            if (i % downsample_rate != 0):
                t += step
                i += 1
                continue
            if t <= t_max:
                lon_pos = lon_coef[0] + lon_coef[1] * t + lon_coef[2] * t ** 2 + lon_coef[3] * t ** 3 + lon_coef[4] * t ** 4
                lat_pos = lat_coef[0] + lat_coef[1] * t + lat_coef[2] * t ** 2 + lat_coef[3] * t ** 3 + lat_coef[4] * t ** 4 + lat_coef[5] * t ** 5
                rel_lon_pos = abs(lon_pos - cur_lon)
                rel_lat_pos = lat_pos - cur_lat if driving_direction == 1 else - (lat_pos - cur_lat)
                trajectory_points.append([rel_lon_pos, rel_lat_pos])
            else:
                rel_lon_pos = trajectory_points[-1][0] + abs(end_lon_v) * step
                rel_lat_pos = trajectory_points[-1][1]
                trajectory_points.append([rel_lon_pos, rel_lat_pos])
            t += step
            i += 1
        return trajectory_points
    
    '''
    description: 
    param {*} self
    param {*} lon_coef 纵向轨迹多项式系数
    param {*} lat_coef 横向轨迹多项式系数
    param {*} t_max 最大时间
    return {*} 是否符合运动学约束
    '''
    def check_kinematic_feasibility(self, lon_coef, lat_coef, t_max):
        t = 1 / SAMPLE_RATE
        step = 1 / SAMPLE_RATE
        while (t <= (t_max + 0.001)):
            vx = derivative(list(lon_coef.values())[::-1], 1, t)
            ax = derivative(list(lon_coef.values())[::-1], 2, t)
            vy = derivative(list(lat_coef.values())[::-1], 1, t)
            ay = derivative(list(lat_coef.values())[::-1], 2, t)
            #计算曲率
            k = abs(vx * ay - ax * vy) / ((vx ** 2 + vy ** 2) ** 1.5)
            if k >= self.k_max:
                return False
            t += step
        return True
    
    
    
    '''
    description: 计算两条轨迹的欧式距离
    param {*} traj1
    param {*} traj2
    return {*}
    '''
    # def compute_trajectory_distance(self, traj1, traj2):
    #     assert len(traj1) == len(traj2), "两条轨迹的时间步数应相同"
        
    #     distance = 0.0
    #     for p1, p2 in zip(traj1, traj2):
    #         distance += np.linalg.norm(np.array(p1) - np.array(p2))
        
    #     return distance / len(traj1)
    
    def compute_trajectory_distance(self, traj1, traj2):
        assert len(traj1) == len(traj2), "两条轨迹的时间步数应相同"
        
        # 转换为numpy数组
        traj1_array = np.array(traj1)
        traj2_array = np.array(traj2)
        
        # 计算欧氏距离并求平均
        return np.mean(np.linalg.norm(traj1_array - traj2_array, axis=1))
    
    
    
    '''
    description: 使用NMS对轨迹进行筛选
    param {*} self
    param {*} trajectories
    param {*} distance_threshold
    return {*}
    '''
    def nms_on_trajectories(self, trajectories, distance_threshold):
        selected_trajectories = []
        
        for traj1 in trajectories:
            # 判断当前轨迹与已选择的轨迹之间是否有重叠
            is_redundant = any(self.compute_trajectory_distance(traj1, traj2) < distance_threshold for traj2 in selected_trajectories)
            
            if not is_redundant:
                selected_trajectories.append(traj1)  # 将不冗余的轨迹加入选择列表
                
        return selected_trajectories





    
            
class TrajectoryDecoder(nn.Module):
    def __init__(self, input_size, driving_style_hidden_size, hidden_size, num_layers, n_predict, use_traj_prior = False, use_endpoint_prior = False) -> None:
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
                nn.LeakyReLU(0.1),
                nn.Linear(self.hidden_size, 1, dtype=torch.float64),
            )
        #自注意力机制
        self.mode2mode_att = nn.MultiheadAttention(embed_dim=self.input_size, num_heads=1, batch_first=True, dtype=torch.float64)
        if use_traj_prior or use_endpoint_prior:
            if use_traj_prior and not use_endpoint_prior:
                self.driving_style_generator = nn.LSTM(input_size=2, 
                                                       hidden_size=self.driving_style_hidden_size, 
                                                       num_layers=num_layers, 
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
    '''
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
        obs_feature = obs_feature.unsqueeze(1).expand(-1, self.n_predict, -1)
        lane_change_feature = lane_change_feature.unsqueeze(1).expand(-1, self.n_predict, -1)
        input_ = torch.cat([obs_feature, lane_change_feature, driving_style], dim=-1)
        input_, _= self.mode2mode_att(input_, input_, input_)
        lstm_input = input_.contiguous().view(bs * self.n_predict, 1, -1).expand(-1, out_length, -1)
        traj_output, _ = self.lstm_layer(lstm_input)
        traj_output = traj_output.view(bs, self.n_predict, out_length, self.hidden_size)
        traj_output = self.traj_head(traj_output)
        traj_prob_output = self.trajprob_head(input_).squeeze(-1)
        return traj_output, traj_prob_output, traj_output[:, :, -1, :]



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
    

class AnchorBasedTrajectoryDecoder(nn.Module):
    def __init__(self, input_dim, driving_style_hidden_size, hidden_dim, num_layers, n_predict) -> None:
      super(AnchorBasedTrajectoryDecoder, self).__init__()
      self.input_dim = input_dim
      self.driving_style_hidden_size = driving_style_hidden_size
      self.hidden_dim = hidden_dim
      self.n_predict = n_predict
      self.driving_style = nn.Parameter(torch.randn(1, n_predict, driving_style_hidden_size, dtype=torch.float64))
      self.hidden_layer = nn.Sequential(
          nn.Linear(input_dim, hidden_dim, dtype=torch.float64),
          nn.LeakyReLU(0.1),
          )
      self.endpoint_predictor = nn.Sequential(
          nn.Linear(hidden_dim, 2, dtype=torch.float64),
          )
      self.confidence_predictor = nn.Sequential(
          nn.Linear(hidden_dim, 1, dtype=torch.float64),
          )
      self.trajectory_decoder = nn.LSTM(input_size=hidden_dim + 2, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dtype=torch.float64)
      self.op = nn.Linear(self.hidden_dim, 2, dtype=torch.float64)
    
    def forward(self, obs_feature, lane_change_feature, out_length):
        # 输入验证
        if not isinstance(out_length, int) or out_length <= 0:
            raise ValueError("out_length 必须是正整数")
        
        bs = obs_feature.shape[0]
        if len(obs_feature.shape) != 2 or len(lane_change_feature.shape) != 2:
            raise ValueError("obs_feature 和 lane_change_feature 必须是二维张量")

        # 验证 driving_style 的形状
        if self.driving_style.shape[0] != 1:
            raise ValueError("driving_style 的第一维必须为 1")

        # 广播机制代替 unsqueeze 和 repeat
        driving_style = self.driving_style.expand(bs, self.n_predict, -1)
        obs_feature_expanded = obs_feature.unsqueeze(1).expand(-1, self.n_predict, -1)
        lane_change_feature_expanded = lane_change_feature.unsqueeze(1).expand(-1, self.n_predict, -1)

        # 合并特征
        combined_features = torch.cat([obs_feature_expanded, lane_change_feature_expanded, driving_style], dim=-1)

        # 隐藏层处理
        hidden_state = self.hidden_layer(combined_features)

        # 预测终点和置信度
        endpoint_output = self.endpoint_predictor(hidden_state)
        confidence_output = self.confidence_predictor(hidden_state).squeeze(-1)

        # 解码器输入准备
        decoder_input = torch.cat([hidden_state, endpoint_output], dim=-1)  # (bs, n_predict, hidden_dim + 2)

        # LSTM 输入准备
        lstm_input = decoder_input.view(bs * self.n_predict, 1, -1).expand(-1, out_length, -1)

        # 轨迹解码
        traj_output, _ = self.trajectory_decoder(lstm_input)
        traj_output = traj_output.view(bs, self.n_predict, out_length, self.hidden_dim)

        # 最终输出
        final_output = self.op(traj_output)

        return final_output, confidence_output, endpoint_output

def trajectory_generator_by_torch(init_cond, lanes_info, target_lane, target_time, n_pred, dt):
    """
    生成未来轨迹 (bs, n_pred, 2)，支持每个样本单独的目标到达时间
    
    Args:
        init_cond: 初始条件 (bs, 6), [x, y, vx, vy, ax, ay]
        lane_info: 车道信息 (bs, 3), [左车道中心y, 直行车道中心y, 右车道中心y]
        target_lane: 3_dim one-hots (3, )
        target_time: (bs,6), 每个样本的目标到达时间
        n_pred: 时间步数
    
    Returns:
        trajectory: (bs, n_time, n_pred, 2), [x, y] 轨迹
    """ 
    
    bs = init_cond.shape[0]
    n_time = target_time.shape[1]
    x0, y0, vx0, vy0, ax0, ay0 = init_cond[:, 0], init_cond[:, 1], init_cond[:, 2], init_cond[:, 3], init_cond[:, 4], init_cond[:, 5]
    
    # 生成时间序列 (bs, 6, n_pred)
    time_steps = torch.linspace(0, n_pred * dt, n_pred + 1, device=init_cond.device).unsqueeze(0).unsqueeze(0).repeat(bs, n_time, 1)  # (bs, 6, n_pred)
    # time_steps = time_steps * target_time.unsqueeze(-1)  # 按每个样本和目标时间缩放
    #终止条件
    vx1 = vx0.unsqueeze(-1).repeat(1, n_time) + target_time * ax0.unsqueeze(-1).repeat(1, n_time)
    ax1 = ax0.unsqueeze(-1).repeat(1, n_time) 
    y1 = torch.sum(lanes_info * target_lane, dim=1).unsqueeze(-1).repeat(1, n_time)
    #0111修改
    # y1 = torch.sum(lanes_info.unsqueeze(1).repeat(1, n_time, 1) * target_lane, dim=-1)
    current_y = lanes_info[:, 1].unsqueeze(-1).repeat(1, n_time)
    y1 = torch.where(y1 == -1, current_y, y1)
    vy1 = torch.zeros_like(vx1).to(init_cond.device)
    ay1 = torch.zeros_like(ax1).to(init_cond.device)
    #纵向四次多项式轨迹(a0-a4)
    a0 = x0.unsqueeze(-1).repeat(1, n_time)
    a1 = vx0.unsqueeze(-1).repeat(1, n_time)
    a2 = ax0.unsqueeze(-1).repeat(1, n_time) / 2
    b1 = vx1 - a1 - 2 * a2 * target_time
    b2 = ax1 - 2 * a2
    a3 = (3 * b1 - b2 * target_time) / (3 * target_time * target_time)
    a4 = (b2 * target_time - 2 * b1) / (4 * target_time * target_time * target_time)
    
    #横向五次多项式轨迹(c0-c5)
    c0 = y0.unsqueeze(-1).repeat(1, n_time)
    c1 = vy0.unsqueeze(-1).repeat(1, n_time)
    c2 = ay0.unsqueeze(-1).repeat(1, n_time) / 2
    b1 = y1 - c0 - c1 * target_time - c2 * target_time * target_time
    b2 = vy1 - c1 - 2 * c2 * target_time
    b3 = ay1 - 2 * c2
    t_5 = target_time ** 5
    t_4 = target_time ** 4
    t_3 = target_time ** 3
    t_2 = target_time ** 2
    t_1 = target_time
    row1 = torch.stack([t_5, t_4, t_3], dim=-1)
    row2 = torch.stack([5*t_4, 4*t_3, 3*t_2], dim=-1)
    row3 = torch.stack([20*t_3, 12*t_2, 6*t_1], dim=-1)
    A = torch.stack([row1, row2, row3], dim=-2)
    B = torch.cat([b1.unsqueeze(-1).unsqueeze(-1),
                   b2.unsqueeze(-1).unsqueeze(-1),
                   b3.unsqueeze(-1).unsqueeze(-1)], dim=2)
    if (torch.linalg.cond(A) > 1e7).any():
        A_pinv = torch.linalg.pinv(A)
        x = torch.matmul(A_pinv, B)
    else:
        x = torch.linalg.solve(A, B)
    c5 = x[:, :, 0, 0]
    c4 = x[:, :, 1, 0]
    c3 = x[:, :, 2, 0]
    traj_x = a0.unsqueeze(-1) + \
            a1.unsqueeze(-1) * time_steps + \
            a2.unsqueeze(-1) * time_steps * time_steps + \
            a3.unsqueeze(-1) * time_steps * time_steps * time_steps + \
            a4.unsqueeze(-1) * time_steps * time_steps * time_steps * time_steps
    traj_y = c0.unsqueeze(-1) + \
        c1.unsqueeze(-1) * time_steps + \
        c2.unsqueeze(-1) * time_steps * time_steps + \
        c3.unsqueeze(-1) * time_steps * time_steps * time_steps + \
        c4.unsqueeze(-1) * time_steps * time_steps * time_steps * time_steps + \
        c5.unsqueeze(-1) * time_steps * time_steps * time_steps * time_steps * time_steps
    #车辆到达目标车道后做匀加速运动
    x1 = a0 + a1 * t_1 + a2 * t_2 + a3 * t_3 + a4 * t_4
    traj_x_updated, traj_y_updated = replace_invalid_trajectories(traj_x=traj_x,
                                                                  traj_y=traj_y,
                                                                  target_time=target_time,
                                                                  n_pred=n_pred,
                                                                  dt = dt, 
                                                                  x_target= x1,
                                                                  vel_x=vx1,
                                                                  acc_x=ax1,
                                                                  y_target=y1)
    trajectory = torch.stack([traj_x_updated, traj_y_updated], dim=-1)
    # return trajectory
    #减去当前时刻的坐标
    trajectory = trajectory - trajectory[:, :, 0, :].unsqueeze(2)
    driving_direction = (vx0 < 0)
    mask = driving_direction.view(-1, 1, 1, 1).expand(-1, n_time, n_pred + 1, 2)
    trajectory[..., 0] = torch.where(mask[..., 0], -trajectory[..., 0], trajectory[..., 0])
    trajectory[..., 1] = torch.where(~mask[..., 0], -trajectory[..., 1], trajectory[..., 1])
    return trajectory[:, :, 1: , :]



def replace_invalid_trajectories(traj_x, traj_y, target_time, n_pred, dt, x_target, vel_x, acc_x, y_target):
    """
    替换无效的轨迹部分 (x 方向匀加速运动，y 方向保持车辆到达目标车道中心线时的 y 坐标)

    Args:
    traj_x: tensor (bs, n_candidate, n_pred, 2), 车辆在 x 方向上的预测轨迹
    traj_y: tensor (bs, n_candidate, n_pred, 2), 车辆在 y 方向上的预测轨迹
    target_time: tensor (bs, n_candidate), 每条候选轨迹到达目标车道中心线的时间
    n_pred: int, 预测的时间点个数
    dt: float, 时间间隔
    vel_x: tensor (bs, n_candidate), x 方向车辆到达目标车道中心线时的速度
    acc_x: tensor (bs, n_candidate), x 方向车辆到达目标车道中心线时的加速度

    Returns:
    traj_x_updated: tensor (bs, n_candidate, n_pred, 2), 替换后的 x 方向预测轨迹
    traj_y_updated: tensor (bs, n_candidate, n_pred, 2), 替换后的 y 方向预测轨迹
    """
    bs, n_candidate, n_pred= traj_x.shape
    
    # 构造时间戳 (n_pred 个时间点，从 0 到 (n_pred-1)*dt)
    time_stamps = torch.arange(0, round(n_pred * dt, 1), dt, device=traj_x.device).view(1, 1, -1)  # (1, 1, n_pred)
    # 扩展 target_time、vel_x、acc_x 的维度以便广播计算
    target_time_expanded = target_time.unsqueeze(-1)  # (bs, n_candidate, 1)
    vel_x_expanded = vel_x.unsqueeze(-1)  # (bs, n_candidate, 1)
    acc_x_expanded = acc_x.unsqueeze(-1)  # (bs, n_candidate, 1)

    # 找到有效的轨迹点（时间大于等于 target_time 的轨迹点是有效的）
    valid_mask = time_stamps >= target_time_expanded  # (bs, n_candidate, n_pred)
    # 计算 delta_time: 当前时间戳与 target_time 的差值
    delta_time = time_stamps - target_time_expanded  # (bs, n_candidate, n_pred)

    # x 方向匀加速公式计算无效部分的替换值
    x_target = x_target.unsqueeze(-1)  # (bs, n_candidate, 1)，假设第一个时间点的 x 是到达中心线的 x
    x_replace = x_target + vel_x_expanded * delta_time  # (bs, n_candidate, n_pred)

    # y 方向保持车辆到达目标车道中心线时的 y 值
    y_target = y_target.unsqueeze(-1)  # (bs, n_candidate, 1)，假设第一个时间点的 y 是到达中心线的 y

    # 更新 traj_x: 如果是无效点（~valid_mask），用 x_replace 替换
    traj_x_updated = traj_x.clone()
    traj_x_updated = torch.where(valid_mask, x_replace, traj_x)
    # 更新 traj_y: 如果是无效点（~valid_mask），用 y_target 替换
    traj_y_updated = traj_y.clone()
    #test plot
    # test = []
    # for i in range(valid_mask.shape[2]):
    #     if valid_mask[0][0][i]:
    #         test.append(y_target.detach().numpy()[0][0][0])
    #     else:
    #         test.append(traj_y.detach().numpy()[0][0][i])
    traj_y_updated = torch.where(valid_mask, y_target, traj_y)
    return traj_x_updated, traj_y_updated