'''
Author: Yang Jialong
Date: 2024-11-15 10:22:52
LastEditTime: 2024-11-21 10:35:02
Description: 请填写简介
'''
import torch
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
    def generate_future_trajectory(self, target_track_csv, start_frame_idx, target_lane_id):
        predict_frame_idx = start_frame_idx + self.obs_len - 1
        # 1. 获取轨迹的起始状态
        start_state = {}
        start_state['x'] = target_track_csv[X][predict_frame_idx]
        start_state['vx'] = target_track_csv[X_VELOCITY][predict_frame_idx]
        start_state['ax'] = target_track_csv[X_ACCELERATION][predict_frame_idx]
        start_state['y'] = target_track_csv[Y][predict_frame_idx]
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
            if (self.check_kinematic_feasibility(lon_coef, lat_coef, candidate_time)):
                trajectory_points = self.sample_trajectory_from_coef(lon_coef, lat_coef, candidate_time)               
                future_trajectory_list.append(trajectory_points)
        return future_trajectory_list
        
            
        
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
    def sample_trajectory_from_coef(self, lon_coef, lat_coef, t_max):
        trajectory_points = []
        cur_lon = lon_coef[0]
        cur_lat = lat_coef[0]
        end_lon_v = lon_coef[1] + t_max * lon_coef[2] * 2
        driving_direction = 1 if lon_coef[1] < 0 else 2
        t = 1 / SAMPLE_RATE
        step = 1 / SAMPLE_RATE
        i = 1
        while (i <= self.pred_len):
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
            
        