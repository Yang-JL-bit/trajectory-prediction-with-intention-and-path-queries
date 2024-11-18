'''
Author: Yang Jialong
Date: 2024-11-15 10:22:52
LastEditTime: 2024-11-18 11:26:17
Description: 请填写简介
'''
import torch
from dataset.highD.utils import *

class TrajectoryGenerator:
    
    def __init__(self, obs_len, pred_len, lane_info, time_step) -> None:
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.lane_info = lane_info
        self.time_step = time_step
    
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
            future_trajectory = []
            # 3. 获取轨迹的终止状态
            end_state = {}
            end_state['vx'] = start_state['vx'] + candidate_time * start_state['ax']
            end_state['ax'] = start_state['ax']
            end_state['y'] = self.lane_info[target_lane_id]
            end_state['vy'] = 0
            end_state['ay'] = 0
            # 4. 分别获取横向和纵向轨迹多项式系数
            
        
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
    
    
    def get_lat_traj_coef(self, start_state, end_state):
        pass
    
    
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
