'''
Author: Yang Jialong
Date: 2024-11-11 17:33:56
LastEditors: Please set LastEditors
LastEditTime: 2025-02-25 16:43:53
Description: 数据处理
PS: 该部分代码引用了 https://github.com/RobertKrajewski/highD-dataset.git
'''

import pandas as pd
import numpy as np
import torch
import random
import torch.nn.functional as F
import math
import os
from random import sample
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from dataset.highD.utils import *
from torch.utils.data import Dataset, random_split


class HighD(Dataset):
    def __init__(self, raw_data_dir, processed_dir, obs_len, pred_len, process_id = [13], load_id = [13], process_data = False, heading_threshold = 0.01, under_sample = None, traj_sample_rate = None) -> None:
        super().__init__()
        self.raw_data_dir = raw_data_dir
        self.processed_dir = processed_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.heading_threshold = heading_threshold
        self.traj_sample_rate = traj_sample_rate
        self.scene_data_list = []
        if process_data:
            for i in tqdm(process_id, desc='Processing data'):
                print("Processing scene: ", i)
                scene_data = self.generate_training_data(i)
                # self.scene_data_list.extend(scene_data)
                torch.save(scene_data, processed_dir + 'data_{}_0102_3s.pt'.format(i))
            print("数据处理完毕")
        else:
            file_list = os.listdir(processed_dir)
            load_file_list = [file for file in file_list if (any(str(id) in file[0:7] for id in load_id))]
            load_file_list = sorted(load_file_list, key=lambda x: int(x.split('_')[1]))
            #输出所有加载的文件
            print("Will load files: ")
            self.maps_info = {}
            self.dataset_pointer = []
            for file in load_file_list:
                print(file, )
            for file in tqdm(load_file_list):
                scene_data = torch.load(processed_dir + file)
                self.scene_data_list.extend(scene_data)
                if under_sample is not None:
                    print("Under sampling...")
                    self.scene_data_list = self.under_sample_dataset(under_sample)  #每读取一个文件就，降采样一次，要不然会爆内存
            if under_sample is not None:    
                print("Under sampling...")
                self.scene_data_list = self.under_sample_dataset(under_sample)
            print("数据加载完毕")
    def __getitem__(self, id):
        return self.scene_data_list[id]
    
    def __len__(self):
        return len(self.scene_data_list)
    
    def get_driving_direcion_by_pos(self, pos_y, lanes_info):
        #判断该店落在哪条车道上
        lane_num = len(lanes_info)
        lane_id = None
        for id, lane_boundaries in lanes_info.items():
            left_boundary, center_lane, right_boundary = lane_boundaries
            if left_boundary <= pos_y <= right_boundary:
                lane_id = id
                break
        if lane_id is None:
            min_distance = float('inf')
            for id, lane_boundaries in lanes_info.items():
                left_boundary, center_lane, right_boundary = lane_boundaries
                # 计算 pos_y 与车道中心线的距离
                distance = abs(pos_y - center_lane)
                if distance < min_distance:
                    min_distance = distance
                    lane_id = id
        if lane_num == 4:
            return 1 if lane_id < 4 else 2
        elif lane_num == 6:
            return 1 if lane_id < 5 else 2
        elif lane_num == 7:
            return 1 if lane_id < 5 else 2
        
        
    def read_tracks_csv(self, track_csv_path):
        """
        This method reads the tracks file from highD data.

        :param arguments: the parsed arguments for the program containing the input path for the tracks csv file.
        :return: a list containing all tracks as dictionaries.
        """
        # Read the csv file, convert it into a useful data structure
        df = pd.read_csv(track_csv_path)

        # Use groupby to aggregate track info. Less error prone than iterating over the data.
        grouped = df.groupby([TRACK_ID], sort=False)
        # Efficiently pre-allocate an empty list of sufficient size
        tracks = {}
        current_track = 0
        for group_id, rows in grouped:
            bounding_boxes = np.transpose(np.array([rows[X].values,
                                                    rows[Y].values,
                                                    rows[WIDTH].values,
                                                    rows[HEIGHT].values]))
            tracks[np.int64(group_id).item()] = {
                FRAME: rows[FRAME].values,
                X: rows[X].values,
                Y: rows[Y].values,
                BBOX: bounding_boxes,
                WIDTH: rows[WIDTH].values,
                HEIGHT: rows[HEIGHT].values,
                X_VELOCITY: rows[X_VELOCITY].values,
                Y_VELOCITY: rows[Y_VELOCITY].values,
                X_ACCELERATION: rows[X_ACCELERATION].values,
                Y_ACCELERATION: rows[Y_ACCELERATION].values,
                FRONT_SIGHT_DISTANCE: rows[FRONT_SIGHT_DISTANCE].values,
                BACK_SIGHT_DISTANCE: rows[BACK_SIGHT_DISTANCE].values,
                THW: rows[THW].values,
                TTC: rows[TTC].values,
                DHW: rows[DHW].values,
                PRECEDING_X_VELOCITY: rows[PRECEDING_X_VELOCITY].values,
                PRECEDING_ID: rows[PRECEDING_ID].values,
                FOLLOWING_ID: rows[FOLLOWING_ID].values,
                LEFT_FOLLOWING_ID: rows[LEFT_FOLLOWING_ID].values,
                LEFT_ALONGSIDE_ID: rows[LEFT_ALONGSIDE_ID].values,
                LEFT_PRECEDING_ID: rows[LEFT_PRECEDING_ID].values,
                RIGHT_FOLLOWING_ID: rows[RIGHT_FOLLOWING_ID].values,
                RIGHT_ALONGSIDE_ID: rows[RIGHT_ALONGSIDE_ID].values,
                RIGHT_PRECEDING_ID: rows[RIGHT_PRECEDING_ID].values,
                LANE_ID: rows[LANE_ID].values
            }
            current_track = current_track + 1
        return tracks

    def get_lanes_info(self, lane_num, recording_meta):
        lanes_info = {}
        lane_num = len(recording_meta[UPPER_LANE_MARKINGS]) + \
            len(recording_meta[LOWER_LANE_MARKINGS]) - 2
        if lane_num == 4:
            # 4 lanes
            lanes_info[2] = [recording_meta[UPPER_LANE_MARKINGS][0], (recording_meta[UPPER_LANE_MARKINGS][0] + recording_meta[UPPER_LANE_MARKINGS][1]) / 2, recording_meta[UPPER_LANE_MARKINGS][1]]
            lanes_info[3] = [recording_meta[UPPER_LANE_MARKINGS][1], (recording_meta[UPPER_LANE_MARKINGS][1] + recording_meta[UPPER_LANE_MARKINGS][2]) / 2, recording_meta[UPPER_LANE_MARKINGS][2]]
            lanes_info[5] = [recording_meta[LOWER_LANE_MARKINGS][0], (recording_meta[LOWER_LANE_MARKINGS][0] + recording_meta[LOWER_LANE_MARKINGS][1]) / 2, recording_meta[LOWER_LANE_MARKINGS][1]]
            lanes_info[6] = [recording_meta[LOWER_LANE_MARKINGS][1], (recording_meta[LOWER_LANE_MARKINGS][1] + recording_meta[LOWER_LANE_MARKINGS][2]) / 2, recording_meta[LOWER_LANE_MARKINGS][2]]
            # lane_width = ((lanes_info[3] - lanes_info[2]) +
            #             (lanes_info[6] - lanes_info[5])) / 2
        elif lane_num == 6:
            # 6 lanes
            lanes_info[2] = [recording_meta[UPPER_LANE_MARKINGS][0], (recording_meta[UPPER_LANE_MARKINGS][0] + recording_meta[UPPER_LANE_MARKINGS][1]) / 2, recording_meta[UPPER_LANE_MARKINGS][1]]
            lanes_info[3] = [recording_meta[UPPER_LANE_MARKINGS][1], (recording_meta[UPPER_LANE_MARKINGS][1] + recording_meta[UPPER_LANE_MARKINGS][2]) / 2, recording_meta[UPPER_LANE_MARKINGS][2]]
            lanes_info[4] = [recording_meta[UPPER_LANE_MARKINGS][2], (recording_meta[UPPER_LANE_MARKINGS][2] + recording_meta[UPPER_LANE_MARKINGS][3]) / 2, recording_meta[UPPER_LANE_MARKINGS][3]]
            lanes_info[6] = [recording_meta[LOWER_LANE_MARKINGS][0], (recording_meta[LOWER_LANE_MARKINGS][0] + recording_meta[LOWER_LANE_MARKINGS][1]) / 2, recording_meta[LOWER_LANE_MARKINGS][1]]
            lanes_info[7] = [recording_meta[LOWER_LANE_MARKINGS][1], (recording_meta[LOWER_LANE_MARKINGS][1] + recording_meta[LOWER_LANE_MARKINGS][2]) / 2, recording_meta[LOWER_LANE_MARKINGS][2]]
            lanes_info[8] = [recording_meta[LOWER_LANE_MARKINGS][2], (recording_meta[LOWER_LANE_MARKINGS][2] + recording_meta[LOWER_LANE_MARKINGS][3]) / 2, recording_meta[LOWER_LANE_MARKINGS][3]]
            # lane_width = ((lanes_info[3] - lanes_info[2]) + (lanes_info[4] - lanes_info[3]) +
            #             (lanes_info[7] - lanes_info[6]) + (lanes_info[8] - lanes_info[7])) / 4
        elif lane_num == 7:
            # 7 lanes: track 58 ~ 60
            lanes_info[2] = [recording_meta[UPPER_LANE_MARKINGS][0], (recording_meta[UPPER_LANE_MARKINGS][0] + recording_meta[UPPER_LANE_MARKINGS][1]) / 2, recording_meta[UPPER_LANE_MARKINGS][1]]
            lanes_info[3] = [recording_meta[UPPER_LANE_MARKINGS][1], (recording_meta[UPPER_LANE_MARKINGS][1] + recording_meta[UPPER_LANE_MARKINGS][2]) / 2, recording_meta[UPPER_LANE_MARKINGS][2]]
            lanes_info[4] = [recording_meta[UPPER_LANE_MARKINGS][2], (recording_meta[UPPER_LANE_MARKINGS][2] + recording_meta[UPPER_LANE_MARKINGS][3]) / 2, recording_meta[UPPER_LANE_MARKINGS][3]]
            lanes_info[5] = [recording_meta[UPPER_LANE_MARKINGS][3], (recording_meta[UPPER_LANE_MARKINGS][3] + recording_meta[UPPER_LANE_MARKINGS][4]) / 2, recording_meta[UPPER_LANE_MARKINGS][4]]
            lanes_info[7] = [recording_meta[LOWER_LANE_MARKINGS][0], (recording_meta[LOWER_LANE_MARKINGS][0] + recording_meta[LOWER_LANE_MARKINGS][1]) / 2, recording_meta[LOWER_LANE_MARKINGS][1]]
            lanes_info[8] = [recording_meta[LOWER_LANE_MARKINGS][1], (recording_meta[LOWER_LANE_MARKINGS][1] + recording_meta[LOWER_LANE_MARKINGS][2]) / 2, recording_meta[LOWER_LANE_MARKINGS][2]]
            lanes_info[9] = [recording_meta[LOWER_LANE_MARKINGS][2], (recording_meta[LOWER_LANE_MARKINGS][2] + recording_meta[LOWER_LANE_MARKINGS][3]) / 2, recording_meta[LOWER_LANE_MARKINGS][3]]
            # lane_width = ((lanes_info[3] - lanes_info[2]) + (lanes_info[4] - lanes_info[3]) + (
            #     lanes_info[5] - lanes_info[4]) + (lanes_info[8] - lanes_info[7]) + (lanes_info[9] - lanes_info[8])) / 5
        else:
            print("Error: Invalid input!")
        return lanes_info
    
    
    def read_tracks_meta(self, tracks_meta_path):
        """
        This method reads the static info file from highD data.

        :param arguments: the parsed arguments for the program containing the input path for the static csv file.
        :return: the static dictionary - the key is the track_id and the value is the corresponding data for this track
        """
        # Read the csv file, convert it into a useful data structure
        df = pd.read_csv(tracks_meta_path)

        # Declare and initialize the static_dictionary
        static_dictionary = {}

        # Iterate over all rows of the csv because we need to create the bounding boxes for each row
        for i_row in range(df.shape[0]):
            track_id = int(df[TRACK_ID][i_row])
            static_dictionary[track_id] = {TRACK_ID: track_id,
                                        WIDTH: float(df[WIDTH][i_row]),
                                        HEIGHT: float(df[HEIGHT][i_row]),
                                        INITIAL_FRAME: int(df[INITIAL_FRAME][i_row]),
                                        FINAL_FRAME: int(df[FINAL_FRAME][i_row]),
                                        NUM_FRAMES: int(df[NUM_FRAMES][i_row]),
                                        CLASS: str(df[CLASS][i_row]),
                                        DRIVING_DIRECTION: float(df[DRIVING_DIRECTION][i_row]),
                                        TRAVELED_DISTANCE: float(df[TRAVELED_DISTANCE][i_row]),
                                        MIN_X_VELOCITY: float(df[MIN_X_VELOCITY][i_row]),
                                        MAX_X_VELOCITY: float(df[MAX_X_VELOCITY][i_row]),
                                        MEAN_X_VELOCITY: float(df[MEAN_X_VELOCITY][i_row]),
                                        MIN_TTC: float(df[MIN_TTC][i_row]),
                                        MIN_THW: float(df[MIN_THW][i_row]),
                                        MIN_DHW: float(df[MIN_DHW][i_row]),
                                        NUMBER_LANE_CHANGES: int(
                                            df[NUMBER_LANE_CHANGES][i_row])
                                        }
        return static_dictionary


    def read_recording_meta(self, recording_meta_path):
        """
        This method reads the video meta file from highD data.

        :param arguments: the parsed arguments for the program containing the input path for the video meta csv file.
        :return: the meta dictionary containing the general information of the video
        """
        # Read the csv file, convert it into a useful data structure
        df = pd.read_csv(recording_meta_path)

        # Declare and initialize the extracted_meta_dictionary
        extracted_meta_dictionary = {ID: int(df[ID][0]),
                                    FRAME_RATE: int(df[FRAME_RATE][0]),
                                    LOCATION_ID: int(df[LOCATION_ID][0]),
                                    SPEED_LIMIT: float(df[SPEED_LIMIT][0]),
                                    MONTH: str(df[MONTH][0]),
                                    WEEKDAY: str(df[WEEKDAY][0]),
                                    START_TIME: str(df[START_TIME][0]),
                                    DURATION: float(df[DURATION][0]),
                                    TOTAL_DRIVEN_DISTANCE: float(df[TOTAL_DRIVEN_DISTANCE][0]),
                                    TOTAL_DRIVEN_TIME: float(df[TOTAL_DRIVEN_TIME][0]),
                                    N_VEHICLES: int(df[N_VEHICLES][0]),
                                    N_CARS: int(df[N_CARS][0]),
                                    N_TRUCKS: int(df[N_TRUCKS][0]),
                                    UPPER_LANE_MARKINGS: np.fromstring(df[UPPER_LANE_MARKINGS][0], sep=";"),
                                    LOWER_LANE_MARKINGS: np.fromstring(df[LOWER_LANE_MARKINGS][0], sep=";")}
        return extracted_meta_dictionary
    
    
    def get_change_direction(self, lane_num, ori_laneId, new_laneId):
        """
        判断变道方向
        """
        if lane_num == 4:
            if (ori_laneId == 2 and new_laneId == 3) or (ori_laneId == 6 and new_laneId == 5):
                return 1
            else:
                return 2
        else:
            # left:
            if (ori_laneId == 2 and new_laneId == 3) or (ori_laneId == 4 and new_laneId == 5) \
                or (ori_laneId == 3 and new_laneId == 4) or (ori_laneId == 7 and new_laneId == 6) \
                    or (ori_laneId == 8 and new_laneId == 7) or (ori_laneId == 9 and new_laneId == 8):
                return 1
            else:
                return 2
    
    def get_left_laneId(self, lane_num, ori_lane_id):
        if lane_num == 4:
            if ori_lane_id == 2:
                return 3
            elif ori_lane_id == 6:
                return 5
            else:
                return None
        elif lane_num == 6:
            if ori_lane_id == 2:
                return 3
            elif ori_lane_id == 3:
                return 4
            elif ori_lane_id == 8:
                return 7
            elif ori_lane_id == 7:
                return 6
            else:
                return None
        else:
            if ori_lane_id == 2:
                return 3
            elif ori_lane_id == 3:
                return 4
            elif ori_lane_id == 8:
                return 7
            elif ori_lane_id == 7:
                return 6
            elif ori_lane_id == 9:
                return 8
            else:
                return None
        
    def get_right_laneId(self, lane_num, ori_lane_id):
        if lane_num == 4:
            if ori_lane_id == 3:
                return 2
            elif ori_lane_id == 5:
                return 6
            else:
                return None
        elif lane_num == 6:
            if ori_lane_id == 3:
                return 2
            elif ori_lane_id == 4:
                return 3
            elif ori_lane_id == 7:
                return 8
            elif ori_lane_id == 6:
                return 7
            else:
                return None
        else:
            if ori_lane_id == 3:
                return 2
            elif ori_lane_id == 4:
                return 3
            elif ori_lane_id == 7:
                return 8
            elif ori_lane_id == 6:
                return 7
            elif ori_lane_id == 8:
                return 9
            else:
                return None
    
    
    def detect_lane_change(self, v_y, v_x, heading_threshold):
        if v_x != 0:
            yaw = abs(math.atan(v_y / v_x))
        else:
            yaw = math.pi / 2 
        # print("yaw: ", yaw)
        return yaw >= heading_threshold
    
    
    def get_traj_label(self, end_frame_idx, lane_change_info):
        """
        返回轨迹的变道id
        保持直行0
        左变道1
        右变道2
        """
        for i in range(len(lane_change_info)):
            if end_frame_idx >= lane_change_info[i][0] and end_frame_idx < lane_change_info[i][1]:
                return lane_change_info[i][3]
        return 0
        
    
    
    def get_lane_changing_info(self, tracks_df, lane_num):
        """
        获取变道信息
        返回(starting_frame, lane_changing_frame, ending_frame, lane_change_direction)
        """
        lane_change_info = []
        last_boundry = 0
        for i in range(1, len(tracks_df[FRAME])):
            if tracks_df[LANE_ID][i] != tracks_df[LANE_ID][i - 1]:
                original_lane = tracks_df[LANE_ID][i - 1]
                new_lane = tracks_df[LANE_ID][i]
                direction = self.get_change_direction(lane_num, original_lane, new_lane)
                #寻找变道起始点
                starting_change = i - 1
                patience = 0  #连续3个点小于阈值终止查询
                starting_change_temp = starting_change
                while starting_change > last_boundry:
                    if not self.detect_lane_change(tracks_df[Y_VELOCITY][starting_change], tracks_df[X_VELOCITY][starting_change], self.heading_threshold):
                        if (patience == 0):
                            starting_change_temp = starting_change
                        patience += 1
                    else:
                        patience = 0
                        starting_change_temp = starting_change
                    if (patience == 3):
                        starting_change = starting_change_temp
                        break
                    starting_change -= 1
                #寻找变道终止点
                ending_change = i
                patience = 0
                ending_change_temp = ending_change
                while ending_change < len(tracks_df[FRAME]):
                    if not self.detect_lane_change(tracks_df[Y_VELOCITY][ending_change], tracks_df[X_VELOCITY][ending_change], self.heading_threshold):
                        if (patience == 0):
                            ending_change_temp = ending_change
                        patience += 1
                    else:
                        patience = 0
                        ending_change_temp = ending_change
                    if (patience == 3):
                        ending_change = ending_change_temp
                        break
                    ending_change += 1
                lane_change_info.append([starting_change, i, ending_change, direction])
                last_boundry = ending_change
        return lane_change_info
                         
        
    
    def construct_traj_features(self, tracks_csv, tracks_meta, id, start_frame_idx, lanes_info, driving_direction):
        target_feature = []
        target_gt = []
        origin_feature = []
        centerline_info = []
        preceding_feature = []
        following_feature = []
        left_following_feature = []
        left_alongside_feature = []
        left_preceding_feature = []
        right_following_feature = []
        right_alongside_feature = []
        right_preceding_feature = []
        surrounding_feature = []
        target_track_csv = tracks_csv[id]
        target_track_meta = tracks_meta[id]
        predict_frame_idx = start_frame_idx + self.obs_len - 1
        while ((predict_frame_idx - start_frame_idx) % self.traj_sample_rate != 0):
            predict_frame_idx = predict_frame_idx - 1
        i = start_frame_idx
        while (i < start_frame_idx + self.obs_len):
            if ((i - start_frame_idx) % self.traj_sample_rate != 0):
                i += 1
                continue
            # 自身特征
            target_feature_temp = []
            target_feature_temp.append(target_track_csv[X][i] - target_track_csv[X][predict_frame_idx] if driving_direction == 2
                                       else target_track_csv[X][predict_frame_idx] - target_track_csv[X][i])
            target_feature_temp.append(target_track_csv[Y][i] - lanes_info[target_track_csv[LANE_ID][predict_frame_idx]][1] if driving_direction == 1
                                       else lanes_info[target_track_csv[LANE_ID][predict_frame_idx]][1] - target_track_csv[Y][i])
            target_feature_temp.append(target_track_csv[WIDTH][i])
            target_feature_temp.append(target_track_csv[HEIGHT][i])
            target_feature_temp.append(-1 * target_track_csv[X_VELOCITY][i] if driving_direction == 1 else target_track_csv[X_VELOCITY][i])
            target_feature_temp.append(-1 * target_track_csv[Y_VELOCITY][i] if driving_direction == 2 else target_track_csv[Y_VELOCITY][i])
            target_feature_temp.append(-1 * target_track_csv[X_ACCELERATION][i] if driving_direction == 1 else target_track_csv[X_ACCELERATION][i])
            target_feature_temp.append(-1 * target_track_csv[Y_ACCELERATION][i] if driving_direction == 2 else target_track_csv[Y_ACCELERATION][i])
            target_feature_temp.extend([1,0] if target_track_meta[CLASS] == 'Car' else [0,1])
            target_feature.append(target_feature_temp)
    
            # preceding vehicle
            preceding_feature_temp = []
            preceding_id = target_track_csv[PRECEDING_ID][predict_frame_idx]
            if preceding_id != 0:
                preceding_track_csv = tracks_csv[preceding_id]
                preceding_track_meta = tracks_meta[preceding_id]
                if tracks_meta[preceding_id][INITIAL_FRAME] <= tracks_meta[id][INITIAL_FRAME] + i:
                    j = tracks_meta[id][INITIAL_FRAME] + i - tracks_meta[preceding_id][INITIAL_FRAME]
                    preceding_feature_temp.append(preceding_track_csv[X][j] - target_track_csv[X][predict_frame_idx] if driving_direction == 2
                                                else target_track_csv[X][predict_frame_idx] - preceding_track_csv[X][j])
                    preceding_feature_temp.append(preceding_track_csv[Y][j] - target_track_csv[Y][predict_frame_idx] if driving_direction == 1
                                                else target_track_csv[Y][predict_frame_idx] - preceding_track_csv[Y][j])
                    preceding_feature_temp.append(preceding_track_csv[WIDTH][j])
                    preceding_feature_temp.append(preceding_track_csv[HEIGHT][j])
                    preceding_feature_temp.append(-1 * preceding_track_csv[X_VELOCITY][j] if driving_direction == 1 else preceding_track_csv[X_VELOCITY][j])
                    preceding_feature_temp.append(-1 * preceding_track_csv[Y_VELOCITY][j] if driving_direction == 2 else preceding_track_csv[Y_VELOCITY][j])
                    preceding_feature_temp.append(-1 * preceding_track_csv[X_ACCELERATION][j] if driving_direction == 1 else preceding_track_csv[X_ACCELERATION][j])
                    preceding_feature_temp.append(-1 * preceding_track_csv[Y_ACCELERATION][j] if driving_direction == 2 else preceding_track_csv[Y_ACCELERATION][j])
                    preceding_feature_temp.extend([1,0] if preceding_track_meta[CLASS] == 'Car' else [0,1])
                else:
                    preceding_feature_temp.extend([0] * 10)
            else:
                preceding_feature_temp.extend([0] * 10)
            preceding_feature.append(preceding_feature_temp)
            
            # following vehicle
            following_feature_temp = []
            following_id = target_track_csv[FOLLOWING_ID][predict_frame_idx]
            if following_id != 0:
                following_track_csv = tracks_csv[following_id]
                following_track_meta = tracks_meta[following_id]
                if tracks_meta[following_id][INITIAL_FRAME] <= tracks_meta[id][INITIAL_FRAME] + i:
                    j = tracks_meta[id][INITIAL_FRAME] + i - tracks_meta[following_id][INITIAL_FRAME]
                    following_feature_temp.append(following_track_csv[X][j] - target_track_csv[X][predict_frame_idx] if driving_direction == 2
                                                else target_track_csv[X][predict_frame_idx] - following_track_csv[X][j])
                    following_feature_temp.append(following_track_csv[Y][j] - target_track_csv[Y][predict_frame_idx] if driving_direction == 1
                                                else target_track_csv[Y][predict_frame_idx] - following_track_csv[Y][j])
                    following_feature_temp.append(following_track_csv[WIDTH][j])
                    following_feature_temp.append(following_track_csv[HEIGHT][j])
                    following_feature_temp.append(-1 * following_track_csv[X_VELOCITY][j] if driving_direction == 1 else following_track_csv[X_VELOCITY][j])
                    following_feature_temp.append(-1 * following_track_csv[Y_VELOCITY][j] if driving_direction == 2 else following_track_csv[Y_VELOCITY][j])
                    following_feature_temp.append(-1 * following_track_csv[X_ACCELERATION][j] if driving_direction == 1 else following_track_csv[X_ACCELERATION][j])
                    following_feature_temp.append(-1 * following_track_csv[Y_ACCELERATION][j] if driving_direction == 2 else following_track_csv[Y_ACCELERATION][j])
                    following_feature_temp.extend([1,0] if following_track_meta[CLASS] == 'Car' else [0,1])
                else:
                    following_feature_temp.extend([0] * 10)
            else:
                following_feature_temp.extend([0] * 10)
            following_feature.append(following_feature_temp)
            
            
            # left following vehicle
            left_following_feature_temp = []
            left_following_id = target_track_csv[LEFT_FOLLOWING_ID][predict_frame_idx]
            if left_following_id != 0:
                left_following_track_csv = tracks_csv[left_following_id]
                left_following_track_meta = tracks_meta[left_following_id]
                if tracks_meta[left_following_id][INITIAL_FRAME] <= tracks_meta[id][INITIAL_FRAME] + i:
                    j = tracks_meta[id][INITIAL_FRAME] + i - tracks_meta[left_following_id][INITIAL_FRAME]
                    left_following_feature_temp.append(left_following_track_csv[X][j] - target_track_csv[X][predict_frame_idx] if driving_direction == 2
                                                else target_track_csv[X][predict_frame_idx] - left_following_track_csv[X][j])
                    left_following_feature_temp.append(left_following_track_csv[Y][j] - target_track_csv[Y][predict_frame_idx] if driving_direction == 1
                                                else target_track_csv[Y][predict_frame_idx] - left_following_track_csv[Y][j])
                    left_following_feature_temp.append(left_following_track_csv[WIDTH][j])
                    left_following_feature_temp.append(left_following_track_csv[HEIGHT][j])
                    left_following_feature_temp.append(-1 * left_following_track_csv[X_VELOCITY][j] if driving_direction == 1 else left_following_track_csv[X_VELOCITY][j])
                    left_following_feature_temp.append(-1 * left_following_track_csv[Y_VELOCITY][j] if driving_direction == 2 else left_following_track_csv[Y_VELOCITY][j])
                    left_following_feature_temp.append(-1 * left_following_track_csv[X_ACCELERATION][j] if driving_direction == 1 else left_following_track_csv[X_ACCELERATION][j])
                    left_following_feature_temp.append(-1 * left_following_track_csv[Y_ACCELERATION][j] if driving_direction == 2 else left_following_track_csv[Y_ACCELERATION][j])
                    left_following_feature_temp.extend([1,0] if left_following_track_meta[CLASS] == 'Car' else [0,1])
                else:
                    left_following_feature_temp.extend([0] * 10)
            else:
                left_following_feature_temp.extend([0] * 10)
            left_following_feature.append(left_following_feature_temp)
            
            # left alongside vehicle
            left_alongside_feature_temp = []
            left_alongside_id = target_track_csv[LEFT_ALONGSIDE_ID][predict_frame_idx]
            if left_alongside_id != 0:
                left_alongside_track_csv = tracks_csv[left_alongside_id]
                left_alongside_track_meta = tracks_meta[left_alongside_id]
                if tracks_meta[left_alongside_id][INITIAL_FRAME] <= tracks_meta[id][INITIAL_FRAME] + i:
                    j = tracks_meta[id][INITIAL_FRAME] + i - tracks_meta[left_alongside_id][INITIAL_FRAME]
                    left_alongside_feature_temp.append(left_alongside_track_csv[X][j] - target_track_csv[X][predict_frame_idx] if driving_direction == 2
                                                else target_track_csv[X][predict_frame_idx] - left_alongside_track_csv[X][j])
                    left_alongside_feature_temp.append(left_alongside_track_csv[Y][j] - target_track_csv[Y][predict_frame_idx] if driving_direction == 1
                                                else target_track_csv[Y][predict_frame_idx] - left_alongside_track_csv[Y][j])
                    left_alongside_feature_temp.append(left_alongside_track_csv[WIDTH][j])
                    left_alongside_feature_temp.append(left_alongside_track_csv[HEIGHT][j])
                    left_alongside_feature_temp.append(-1 * left_alongside_track_csv[X_VELOCITY][j] if driving_direction == 1 else left_alongside_track_csv[X_VELOCITY][j])
                    left_alongside_feature_temp.append(-1 * left_alongside_track_csv[Y_VELOCITY][j] if driving_direction == 2 else left_alongside_track_csv[Y_VELOCITY][j])
                    left_alongside_feature_temp.append(-1 * left_alongside_track_csv[X_ACCELERATION][j] if driving_direction == 1 else left_alongside_track_csv[X_ACCELERATION][j])
                    left_alongside_feature_temp.append(-1 * left_alongside_track_csv[Y_ACCELERATION][j] if driving_direction == 2 else left_alongside_track_csv[Y_ACCELERATION][j])
                    left_alongside_feature_temp.extend([1,0] if left_alongside_track_meta[CLASS] == 'Car' else [0,1])
                else:
                    left_alongside_feature_temp.extend([0] * 10)
            else:
                left_alongside_feature_temp.extend([0] * 10)
            left_alongside_feature.append(left_alongside_feature_temp)
            
            # left preceding vehicle
            left_preceding_feature_temp = []
            left_preceding_id = target_track_csv[LEFT_PRECEDING_ID][predict_frame_idx]
            if left_preceding_id != 0:
                left_preceding_track_csv = tracks_csv[left_preceding_id]
                left_preceding_track_meta = tracks_meta[left_preceding_id]
                if tracks_meta[left_preceding_id][INITIAL_FRAME] <= tracks_meta[id][INITIAL_FRAME] + i:
                    j = tracks_meta[id][INITIAL_FRAME] + i - tracks_meta[left_preceding_id][INITIAL_FRAME]
                    left_preceding_feature_temp.append(left_preceding_track_csv[X][j] - target_track_csv[X][predict_frame_idx] if driving_direction == 2
                                                else target_track_csv[X][predict_frame_idx] - left_preceding_track_csv[X][j])
                    left_preceding_feature_temp.append(left_preceding_track_csv[Y][j] - target_track_csv[Y][predict_frame_idx] if driving_direction == 1
                                                else target_track_csv[Y][predict_frame_idx] - left_preceding_track_csv[Y][j])
                    left_preceding_feature_temp.append(left_preceding_track_csv[WIDTH][j])
                    left_preceding_feature_temp.append(left_preceding_track_csv[HEIGHT][j])
                    left_preceding_feature_temp.append(-1 * left_preceding_track_csv[X_VELOCITY][j] if driving_direction == 1 else left_preceding_track_csv[X_VELOCITY][j])
                    left_preceding_feature_temp.append(-1 * left_preceding_track_csv[Y_VELOCITY][j] if driving_direction == 2 else left_preceding_track_csv[Y_VELOCITY][j])
                    left_preceding_feature_temp.append(-1 * left_preceding_track_csv[X_ACCELERATION][j] if driving_direction == 1 else left_preceding_track_csv[X_ACCELERATION][j])
                    left_preceding_feature_temp.append(-1 * left_preceding_track_csv[Y_ACCELERATION][j] if driving_direction == 2 else left_preceding_track_csv[Y_ACCELERATION][j])
                    left_preceding_feature_temp.extend([1,0] if left_preceding_track_meta[CLASS] == 'Car' else [0,1])
                else:
                    left_preceding_feature_temp.extend([0] * 10)
            else:
                left_preceding_feature_temp.extend([0] * 10)
            left_preceding_feature.append(left_preceding_feature_temp)
            
            # right following id
            right_following_feature_temp = []
            right_following_id = target_track_csv[RIGHT_FOLLOWING_ID][predict_frame_idx]
            if right_following_id != 0:
                right_following_track_csv = tracks_csv[right_following_id]
                right_following_track_meta = tracks_meta[right_following_id]
                if tracks_meta[right_following_id][INITIAL_FRAME] <= tracks_meta[id][INITIAL_FRAME] + i:
                    j = tracks_meta[id][INITIAL_FRAME] + i - tracks_meta[right_following_id][INITIAL_FRAME]
                    right_following_feature_temp.append(right_following_track_csv[X][j] - target_track_csv[X][predict_frame_idx] if driving_direction == 2
                                                else target_track_csv[X][predict_frame_idx] - right_following_track_csv[X][j])
                    right_following_feature_temp.append(right_following_track_csv[Y][j] - target_track_csv[Y][predict_frame_idx] if driving_direction == 1
                                                else target_track_csv[Y][predict_frame_idx] - right_following_track_csv[Y][j])
                    right_following_feature_temp.append(right_following_track_csv[WIDTH][j])
                    right_following_feature_temp.append(right_following_track_csv[HEIGHT][j])
                    right_following_feature_temp.append(-1 * right_following_track_csv[X_VELOCITY][j] if driving_direction == 1 else right_following_track_csv[X_VELOCITY][j])
                    right_following_feature_temp.append(-1 * right_following_track_csv[Y_VELOCITY][j] if driving_direction == 2 else right_following_track_csv[Y_VELOCITY][j])
                    right_following_feature_temp.append(-1 * right_following_track_csv[X_ACCELERATION][j] if driving_direction == 1 else right_following_track_csv[X_ACCELERATION][j])
                    right_following_feature_temp.append(-1 * right_following_track_csv[Y_ACCELERATION][j] if driving_direction == 2 else right_following_track_csv[Y_ACCELERATION][j])
                    right_following_feature_temp.extend([1,0] if right_following_track_meta[CLASS] == 'Car' else [0,1])
                else:
                    right_following_feature_temp.extend([0] * 10)
            else:
                right_following_feature_temp.extend([0] * 10)
            right_following_feature.append(right_following_feature_temp)
            
            # right alongside id
            right_alongside_feature_temp = []
            right_alongside_id = target_track_csv[RIGHT_ALONGSIDE_ID][predict_frame_idx]
            if right_alongside_id != 0:
                right_alongside_track_csv = tracks_csv[right_alongside_id]
                right_alongside_track_meta = tracks_meta[right_alongside_id]
                if tracks_meta[right_alongside_id][INITIAL_FRAME] <= tracks_meta[id][INITIAL_FRAME] + i:
                    j = tracks_meta[id][INITIAL_FRAME] + i - tracks_meta[right_alongside_id][INITIAL_FRAME]
                    right_alongside_feature_temp.append(right_alongside_track_csv[X][j] - target_track_csv[X][predict_frame_idx] if driving_direction == 2
                                                else target_track_csv[X][predict_frame_idx] - right_alongside_track_csv[X][j])
                    right_alongside_feature_temp.append(right_alongside_track_csv[Y][j] - target_track_csv[Y][predict_frame_idx] if driving_direction == 1
                                                else target_track_csv[Y][predict_frame_idx] - right_alongside_track_csv[Y][j])
                    right_alongside_feature_temp.append(right_alongside_track_csv[WIDTH][j])
                    right_alongside_feature_temp.append(right_alongside_track_csv[HEIGHT][j])
                    right_alongside_feature_temp.append(-1 * right_alongside_track_csv[X_VELOCITY][j] if driving_direction == 1 else right_alongside_track_csv[X_VELOCITY][j])
                    right_alongside_feature_temp.append(-1 * right_alongside_track_csv[Y_VELOCITY][j] if driving_direction == 2 else right_alongside_track_csv[Y_VELOCITY][j])
                    right_alongside_feature_temp.append(-1 * right_alongside_track_csv[X_ACCELERATION][j] if driving_direction == 1 else right_alongside_track_csv[X_ACCELERATION][j])
                    right_alongside_feature_temp.append(-1 * right_alongside_track_csv[Y_ACCELERATION][j] if driving_direction == 2 else right_alongside_track_csv[Y_ACCELERATION][j])
                    right_alongside_feature_temp.extend([1,0] if right_alongside_track_meta[CLASS] == 'Car' else [0,1])
                else:
                    right_alongside_feature_temp.extend([0] * 10)
            else:
                right_alongside_feature_temp.extend([0] * 10)
            right_alongside_feature.append(right_alongside_feature_temp)

            # right preceding id
            right_preceding_feature_temp = []
            right_preceding_id = target_track_csv[RIGHT_PRECEDING_ID][predict_frame_idx]
            if right_preceding_id != 0:
                right_preceding_track_csv = tracks_csv[right_preceding_id]
                right_preceding_track_meta = tracks_meta[right_preceding_id]
                if tracks_meta[right_preceding_id][INITIAL_FRAME] <= tracks_meta[id][INITIAL_FRAME] + i:
                    j = tracks_meta[id][INITIAL_FRAME] + i - tracks_meta[right_preceding_id][INITIAL_FRAME]
                    right_preceding_feature_temp.append(right_preceding_track_csv[X][j] - target_track_csv[X][predict_frame_idx] if driving_direction == 2
                                                else target_track_csv[X][predict_frame_idx] - right_preceding_track_csv[X][j])
                    right_preceding_feature_temp.append(right_preceding_track_csv[Y][j] - target_track_csv[Y][predict_frame_idx] if driving_direction == 1
                                                else target_track_csv[Y][predict_frame_idx] - right_preceding_track_csv[Y][j])
                    right_preceding_feature_temp.append(right_preceding_track_csv[WIDTH][j])
                    right_preceding_feature_temp.append(right_preceding_track_csv[HEIGHT][j])
                    right_preceding_feature_temp.append(-1 * right_preceding_track_csv[X_VELOCITY][j] if driving_direction == 1 else right_preceding_track_csv[X_VELOCITY][j])
                    right_preceding_feature_temp.append(-1 * right_preceding_track_csv[Y_VELOCITY][j] if driving_direction == 2 else right_preceding_track_csv[Y_VELOCITY][j])
                    right_preceding_feature_temp.append(-1 * right_preceding_track_csv[X_ACCELERATION][j] if driving_direction == 1 else right_preceding_track_csv[X_ACCELERATION][j])
                    right_preceding_feature_temp.append(-1 * right_preceding_track_csv[Y_ACCELERATION][j] if driving_direction == 2 else right_preceding_track_csv[Y_ACCELERATION][j])
                    right_preceding_feature_temp.extend([1,0] if right_preceding_track_meta[CLASS] == 'Car' else [0,1])
                else:
                    right_preceding_feature_temp.extend([0] * 10)
            else:
                right_preceding_feature_temp.extend([0] * 10)
            right_preceding_feature.append(right_preceding_feature_temp)
            i += 1
        
        
        # 获取目标车辆未来轨迹groundtruth
        while (i < start_frame_idx + self.obs_len + self.pred_len):
            if ((i - start_frame_idx - self.obs_len) % self.traj_sample_rate != 0):
                i += 1
                continue
            target_gt_temp = []
            target_gt_temp.append(target_track_csv[X][i] - target_track_csv[X][predict_frame_idx] if driving_direction == 2
                                  else target_track_csv[X][predict_frame_idx] - target_track_csv[X][i])
            target_gt_temp.append(target_track_csv[Y][i] - target_track_csv[Y][predict_frame_idx]if driving_direction == 1
                                  else target_track_csv[Y][predict_frame_idx] - target_track_csv[Y][i])
            i += 1
            target_gt.append(target_gt_temp)
        
        
        # 获取目标车辆预测原点特征
        origin_feature.append(target_track_csv[X][predict_frame_idx] + target_track_csv[WIDTH][predict_frame_idx] / 2)
        origin_feature.append(target_track_csv[Y][predict_frame_idx] + target_track_csv[HEIGHT][predict_frame_idx] / 2)
        origin_feature.append(target_track_csv[X_VELOCITY][predict_frame_idx])
        origin_feature.append(target_track_csv[Y_VELOCITY][predict_frame_idx])
        origin_feature.append(target_track_csv[X_ACCELERATION][predict_frame_idx])
        origin_feature.append(target_track_csv[Y_ACCELERATION][predict_frame_idx])
        
        #获取目标车辆左侧车道、当前车道以及右侧车道的车道中心线坐标
        cur_laneId = target_track_csv[LANE_ID][predict_frame_idx]
        if self.get_left_laneId(len(lanes_info), cur_laneId) is not None:
            centerline_info.append(lanes_info[self.get_left_laneId(len(lanes_info), cur_laneId)][1])
        else:
            centerline_info.append(-1)
        centerline_info.append(lanes_info[cur_laneId][1])
        if self.get_right_laneId(len(lanes_info), cur_laneId) is not None:
            centerline_info.append(lanes_info[self.get_right_laneId(len(lanes_info), cur_laneId)][1])
        else:
            centerline_info.append(-1) 
        
        surrounding_feature.append(preceding_feature)
        surrounding_feature.append(following_feature)
        surrounding_feature.append(left_following_feature)
        surrounding_feature.append(left_alongside_feature)
        surrounding_feature.append(left_preceding_feature)
        surrounding_feature.append(right_following_feature)
        surrounding_feature.append(right_alongside_feature)
        surrounding_feature.append(right_preceding_feature)
        
        return target_feature, surrounding_feature, target_gt, origin_feature, centerline_info
                    
    def construct_future_traj_feature(self, target_csv, start_frame_idx, lane_num, trajectory_generator):
        predict_frame_idx = start_frame_idx + self.obs_len - 1
        current_lane_id = target_csv[LANE_ID][predict_frame_idx]
        future_traj_feature = []
        future_lc_feature = []
        left_laneId = self.get_left_laneId(lane_num, current_lane_id)
        right_laneId = self.get_right_laneId(lane_num, current_lane_id)
        if left_laneId is not None:
            trajectory_feature = trajectory_generator.generate_future_trajectory(target_csv, start_frame_idx, left_laneId, self.traj_sample_rate)
            future_traj_feature.extend(trajectory_feature)
            future_lc_feature.extend([[1., 0., 0.]] * len(trajectory_feature))
        if right_laneId is not None:
            trajectory_feature = trajectory_generator.generate_future_trajectory(target_csv, start_frame_idx, right_laneId, self.traj_sample_rate)
            future_traj_feature.extend(trajectory_feature)
            future_lc_feature.extend([[0., 0., 1.]] * len(trajectory_feature))
        if current_lane_id is not None:
            trajectory_feature = trajectory_generator.generate_future_trajectory(target_csv, start_frame_idx, current_lane_id, self.traj_sample_rate)
            future_traj_feature.extend(trajectory_feature)
            future_lc_feature.extend([[0., 1., 0.]] * len(trajectory_feature))      
        
        # 将future_traj_feature 和 future_lc_feature padding成固定长度(70)
        mask = [1] * len(future_traj_feature) + [0] * (70 - len(future_traj_feature))
        future_traj_feature.extend([[[0., 0.]] * (self.pred_len // self.traj_sample_rate)] * (70 - len(future_traj_feature)))
        future_lc_feature.extend([[0., 0., 0.]] * (70 - len(future_lc_feature)))
         
        return future_traj_feature, future_lc_feature, mask
    
    
    def cal_candidate_trajectory_score(self, candidate_trajectory: torch.Tensor, groundtruth: torch.Tensor, mask):
        distances = torch.norm(candidate_trajectory - groundtruth.unsqueeze(0).repeat(candidate_trajectory.shape[0], 1, 1), dim=2)
        cumulative_error = distances.sum(dim=1)
        valid_cumulative_error = cumulative_error * mask.float()
        valid_cumulative_error[mask == 0] = float('inf')
        scores = F.softmax(-valid_cumulative_error, dim=0)
        scores = scores * mask.float()
        valid_scores_sum = scores.sum()  # 计算有效轨迹的分数总和
        if valid_scores_sum > 0:
            scores = scores / valid_scores_sum  # 归一化
        return scores
    
    
    # def plot_scene(self, lane_info, target_track_csv, start_frame_idx, origin_feature, centerline_info):
    #     target_gt = []
    #     i = start_frame_idx + self.obs_len
    #     while (i < start_frame_idx + self.obs_len + self.pred_len):
    #         if ((i - start_frame_idx - self.obs_len) % self.traj_sample_rate != 0):
    #             i += 1
    #             continue
    #         target_gt_temp = []
    #         target_gt_temp.append(target_track_csv[X][i] + target_track_csv[WIDTH][i] / 2)
    #         target_gt_temp.append(target_track_csv[Y][i] + target_track_csv[HEIGHT][i] / 2)
    #         i += 1
    #         target_gt.append(target_gt_temp)
    #     lanes_line = []
    #     start_x = target_track_csv[X][start_frame_idx + self.obs_len] - 150
    #     end_x = target_track_csv[X][start_frame_idx + self.obs_len] + 150
    #     line_space_x = np.linspace(start_x, end_x, 200)
    #     traj_pred = trajectory_generator_by_torch(origin_feature, centerline_info, 
    #                                               torch.tensor([[0,1, 0]]), 
    #                                               torch.linspace(0.5 ,10, 20, dtype=torch.float64).unsqueeze(0),
    #                                               n_pred=15, dt = 0.2)[0]
    #     fig, ax = plt.subplots()
    #     for lane_id, laneline in lane_info.items():
    #         ax.plot(line_space_x, np.ones_like(line_space_x) * laneline[0], color = 'black')
    #         ax.plot(line_space_x, np.ones_like(line_space_x) * laneline[2], color = 'black')
    #     ax.plot(np.array(target_gt)[:, 0], np.array(target_gt)[:, 1], color = 'red')
    #     for traj in traj_pred:
    #         ax.plot(traj[:, 0], traj[:, 1], color = 'green')
    #     ax.invert_yaxis()
    #     plt.show()
    
    def generate_training_data(self, number):
        tracks_csv = self.read_tracks_csv(self.raw_data_dir + str(number).zfill(2) + "_tracks.csv")
        tracks_meta = self.read_tracks_meta(self.raw_data_dir + str(number).zfill(2) + "_tracksMeta.csv")
        recording_meta = self.read_recording_meta(self.raw_data_dir + str(number).zfill(2) + "_recordingMeta.csv")
        lanes_info = {}
        lane_num = len(recording_meta[UPPER_LANE_MARKINGS]) + \
            len(recording_meta[LOWER_LANE_MARKINGS]) - 2
        if lane_num == 4:
            # 4 lanes
            lanes_info[2] = [recording_meta[UPPER_LANE_MARKINGS][0], (recording_meta[UPPER_LANE_MARKINGS][0] + recording_meta[UPPER_LANE_MARKINGS][1]) / 2, recording_meta[UPPER_LANE_MARKINGS][1]]
            lanes_info[3] = [recording_meta[UPPER_LANE_MARKINGS][1], (recording_meta[UPPER_LANE_MARKINGS][1] + recording_meta[UPPER_LANE_MARKINGS][2]) / 2, recording_meta[UPPER_LANE_MARKINGS][2]]
            lanes_info[5] = [recording_meta[LOWER_LANE_MARKINGS][0], (recording_meta[LOWER_LANE_MARKINGS][0] + recording_meta[LOWER_LANE_MARKINGS][1]) / 2, recording_meta[LOWER_LANE_MARKINGS][1]]
            lanes_info[6] = [recording_meta[LOWER_LANE_MARKINGS][1], (recording_meta[LOWER_LANE_MARKINGS][1] + recording_meta[LOWER_LANE_MARKINGS][2]) / 2, recording_meta[LOWER_LANE_MARKINGS][2]]
            # lane_width = ((lanes_info[3] - lanes_info[2]) +
            #             (lanes_info[6] - lanes_info[5])) / 2
        elif lane_num == 6:
            # 6 lanes
            lanes_info[2] = [recording_meta[UPPER_LANE_MARKINGS][0], (recording_meta[UPPER_LANE_MARKINGS][0] + recording_meta[UPPER_LANE_MARKINGS][1]) / 2, recording_meta[UPPER_LANE_MARKINGS][1]]
            lanes_info[3] = [recording_meta[UPPER_LANE_MARKINGS][1], (recording_meta[UPPER_LANE_MARKINGS][1] + recording_meta[UPPER_LANE_MARKINGS][2]) / 2, recording_meta[UPPER_LANE_MARKINGS][2]]
            lanes_info[4] = [recording_meta[UPPER_LANE_MARKINGS][2], (recording_meta[UPPER_LANE_MARKINGS][2] + recording_meta[UPPER_LANE_MARKINGS][3]) / 2, recording_meta[UPPER_LANE_MARKINGS][3]]
            lanes_info[6] = [recording_meta[LOWER_LANE_MARKINGS][0], (recording_meta[LOWER_LANE_MARKINGS][0] + recording_meta[LOWER_LANE_MARKINGS][1]) / 2, recording_meta[LOWER_LANE_MARKINGS][1]]
            lanes_info[7] = [recording_meta[LOWER_LANE_MARKINGS][1], (recording_meta[LOWER_LANE_MARKINGS][1] + recording_meta[LOWER_LANE_MARKINGS][2]) / 2, recording_meta[LOWER_LANE_MARKINGS][2]]
            lanes_info[8] = [recording_meta[LOWER_LANE_MARKINGS][2], (recording_meta[LOWER_LANE_MARKINGS][2] + recording_meta[LOWER_LANE_MARKINGS][3]) / 2, recording_meta[LOWER_LANE_MARKINGS][3]]
            # lane_width = ((lanes_info[3] - lanes_info[2]) + (lanes_info[4] - lanes_info[3]) +
            #             (lanes_info[7] - lanes_info[6]) + (lanes_info[8] - lanes_info[7])) / 4
        elif lane_num == 7:
            # 7 lanes: track 58 ~ 60
            lanes_info[2] = [recording_meta[UPPER_LANE_MARKINGS][0], (recording_meta[UPPER_LANE_MARKINGS][0] + recording_meta[UPPER_LANE_MARKINGS][1]) / 2, recording_meta[UPPER_LANE_MARKINGS][1]]
            lanes_info[3] = [recording_meta[UPPER_LANE_MARKINGS][1], (recording_meta[UPPER_LANE_MARKINGS][1] + recording_meta[UPPER_LANE_MARKINGS][2]) / 2, recording_meta[UPPER_LANE_MARKINGS][2]]
            lanes_info[4] = [recording_meta[UPPER_LANE_MARKINGS][2], (recording_meta[UPPER_LANE_MARKINGS][2] + recording_meta[UPPER_LANE_MARKINGS][3]) / 2, recording_meta[UPPER_LANE_MARKINGS][3]]
            lanes_info[5] = [recording_meta[UPPER_LANE_MARKINGS][3], (recording_meta[UPPER_LANE_MARKINGS][3] + recording_meta[UPPER_LANE_MARKINGS][4]) / 2, recording_meta[UPPER_LANE_MARKINGS][4]]
            lanes_info[7] = [recording_meta[LOWER_LANE_MARKINGS][0], (recording_meta[LOWER_LANE_MARKINGS][0] + recording_meta[LOWER_LANE_MARKINGS][1]) / 2, recording_meta[LOWER_LANE_MARKINGS][1]]
            lanes_info[8] = [recording_meta[LOWER_LANE_MARKINGS][1], (recording_meta[LOWER_LANE_MARKINGS][1] + recording_meta[LOWER_LANE_MARKINGS][2]) / 2, recording_meta[LOWER_LANE_MARKINGS][2]]
            lanes_info[9] = [recording_meta[LOWER_LANE_MARKINGS][2], (recording_meta[LOWER_LANE_MARKINGS][2] + recording_meta[LOWER_LANE_MARKINGS][3]) / 2, recording_meta[LOWER_LANE_MARKINGS][3]]
            # lane_width = ((lanes_info[3] - lanes_info[2]) + (lanes_info[4] - lanes_info[3]) + (
            #     lanes_info[5] - lanes_info[4]) + (lanes_info[8] - lanes_info[7]) + (lanes_info[9] - lanes_info[8])) / 5
        else:
            print("Error: Invalid input -", number)
        
        scene_list = []
        #1. 找到变道轨迹和直行轨迹id
        lane_changing_ids = []
        lane_keeping_ids = []
        for key in tracks_meta:
            if tracks_meta[key][NUMBER_LANE_CHANGES] > 0:
                lane_changing_ids.append(key)
            else:
                lane_keeping_ids.append(key)
        # traj_generator = TrajectoryGenerator(self.obs_len, self.pred_len, lanes_info)
        #2. 获取变道轨迹的特征
        for id in tqdm(lane_changing_ids):
            lane_change_info = self.get_lane_changing_info(tracks_csv[id], lane_num)
            driving_direction = tracks_meta[id][DRIVING_DIRECTION]
            for i in range(len(tracks_csv[id][FRAME]) - self.obs_len - self.pred_len + 1):
                target_feature, surroungding_feature, target_gt, origin_feature, centerline_info = self.construct_traj_features(tracks_csv, tracks_meta, id, i, lanes_info, driving_direction)
                # future_traj_feature, future_lc_feature, mask = self.construct_future_traj_feature(tracks_csv[id], i, lane_num, traj_generator)
                lane_change_label = self.get_traj_label(i + self.obs_len - 1, lane_change_info)
                scene_dict = {}
                scene_dict["target_obs_traj"] = torch.tensor(target_feature) #(obs_len, feature_dim)
                scene_dict["surrounding_obs_traj"] = torch.tensor(surroungding_feature) #(8, obs_len, feature_dim)
                scene_dict["lane_change_label"] = torch.tensor(lane_change_label) #(1, )
                scene_dict["future_traj_gt"] = torch.tensor(target_gt) #(pred_len, 2)
                scene_dict["origin_feature"] = torch.tensor(origin_feature) #(6,)
                scene_dict["centerline_info"] = torch.tensor(centerline_info)
                # self.plot_scene(lanes_info, tracks_csv[id], i, 
                #                 scene_dict["origin_feature"].unsqueeze(0), scene_dict["centerline_info"].unsqueeze(0))
                # scene_dict["future_traj_pred"] = torch.tensor(future_traj_feature) # (70, pred_len, 2)
                # scene_dict["future_traj_intention"] = torch.tensor(future_lc_feature) # (70, 3)
                # scene_dict["future_traj_mask"] = torch.tensor(mask) #(70, )
                
                # scene_dict["future_traj_score"] = self.cal_candidate_trajectory_score(scene_dict["future_traj_pred"], scene_dict["future_traj_gt"], scene_dict["future_traj_mask"])
                scene_list.append(scene_dict)
                # for i in range(scene_dict["future_traj_pred"].shape[0]):
                #     color = None
                #     if scene_dict["future_traj_intention"][i, 0] == 1:
                #         color = 'blue'
                #     elif scene_dict["future_traj_intention"][i, 1] == 1:
                #         color = 'red'
                #     elif scene_dict["future_traj_intention"][i, 2] == 1:
                #         color = 'green'
                #     plt.plot(scene_dict["future_traj_pred"][i, :, 0], scene_dict["future_traj_pred"][i, :, 1], color = color)
                # plt.plot(np.array(scene_dict["future_traj_gt"])[:, 0], np.array(scene_dict["future_traj_gt"])[:, 1], color = 'black')
                # plt.show()
        #3. 获取直行轨迹的特征(直行轨迹太多了，只抽一部分就行了)
        if len(lane_keeping_ids) > 100:
            random.seed(12345)
            lane_keeping_ids = sample(lane_keeping_ids, 100)
            
        for id in tqdm(lane_keeping_ids):
            driving_direction = tracks_meta[id][DRIVING_DIRECTION]
            for i in range(len(tracks_csv[id][FRAME]) - self.obs_len - self.pred_len + 1):
                target_feature, surroungding_feature, target_gt, origin_feature, centerline_info = self.construct_traj_features(tracks_csv, tracks_meta, id, i, lanes_info, driving_direction)
                # future_traj_feature, future_lc_feature, mask = self.construct_future_traj_feature(tracks_csv[id], i, lane_num, traj_generator)
                lane_change_label = 0.
                scene_dict = {}
                scene_dict["target_obs_traj"] = torch.tensor(target_feature) #(obs_len, feature_dim)
                scene_dict["surrounding_obs_traj"] = torch.tensor(surroungding_feature) #(8, obs_len, feature_dim)
                scene_dict["lane_change_label"] = torch.tensor(lane_change_label) #(1, )
                scene_dict["future_traj_gt"] = torch.tensor(target_gt) #(pred_len, 2)
                scene_dict["origin_feature"] = torch.tensor(origin_feature)
                scene_dict["centerline_info"] = torch.tensor(centerline_info)
                # scene_dict["future_traj_pred"] = torch.tensor(future_traj_feature) # (70, pred_len, 2)
                # scene_dict["future_traj_intention"] = torch.tensor(future_lc_feature) # (70, 3)
                # scene_dict["future_traj_mask"] = torch.tensor(mask) #(70, )
                scene_list.append(scene_dict)
        return scene_list
    
    def under_sample_dataset(self, n):
        random.seed(42)
        class_0 = [data for data in self.scene_data_list if data['lane_change_label'] == 0]
        class_1 = [data for data in self.scene_data_list if data['lane_change_label'] == 1]
        class_2 = [data for data in self.scene_data_list if data['lane_change_label'] == 2]
        min_samples = min(len(class_0), len(class_1), len(class_2), n)
        print(f'Min samples: {min_samples}')
        class_0_sampled = random.sample(class_0, min_samples)
        class_1_sampled = random.sample(class_1, min_samples)    
        class_2_sampled = random.sample(class_2, min_samples)
        return class_0_sampled + class_1_sampled + class_2_sampled
def get_label_weight(scene_list):
    label_num = [0, 0, 0]
    for scene in scene_list:
        if scene['lane_change_label'] == 0:
            label_num[0] += 1
        elif scene['lane_change_label'] == 1:
            label_num[1] += 1
        else:
            label_num[2] += 1
    print(label_num)
    return [sum(label_num) / i for i in label_num]

def standard_normalization(input_tensor):
    if len(input_tensor.shape) == 3:
        # 假设输入tensor是input_tensor，维度为(batch_size, time_step, feature_dim)
        batch_size, time_step, feature_dim = input_tensor.shape
        
        # 计算每个特征的均值和标准差
        mean = input_tensor.mean(dim=(0, 1))  # 在batch和时间步上求均值
        std = input_tensor.std(dim=(0, 1), unbiased=False)  # 在batch和时间步上求标准差
        
        # 避免除以零的情况
        std[std == 0] = 1
    elif len(input_tensor.shape) == 4:
        mean = input_tensor.mean(dim=(0, 2))  # 在batch和时间步上求均值
        std = input_tensor.std(dim=(0, 2), unbiased=False)  # 在batch和时间步上求标准差
        
        # 避免除以零的情况
        std[std == 0] = 1
    return [mean, std]

    

def get_sample_weight(dataset, label_weight):
    sample_weight = []
    for i in range(len(dataset)):
        if dataset[i]['lane_change_label'] == 0:
            sample_weight.append(label_weight[0])
        elif dataset[i]['lane_change_label'] == 1:
            sample_weight.append(label_weight[1])
        else:
            sample_weight.append(label_weight[2])
    return torch.tensor(sample_weight)

def split_dataset(dataset):
    """
    按照(0.8, 0.1, 0.1)比例划分数据集（训练集，测试集，验证集）
    """
    seed = 42
    torch.manual_seed(seed)
    train_size = int(0.8 * len(dataset))
    temp_size = len(dataset) - train_size
    val_size = int(0.5 * temp_size)
    test_size = temp_size - val_size
    
    # 第一次划分为训练集和临时集
    train_dataset, temp_dataset = random_split(dataset, [train_size, temp_size])

    # 第二次划分临时集为验证集和测试集
    val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])
    
    return train_dataset, val_dataset, test_dataset