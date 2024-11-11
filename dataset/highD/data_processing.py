
import pandas as pd
import numpy as np
import torch
import math
from tqdm import tqdm, trange

from dataset.highD.utils import *
from torch.utils.data import Dataset, random_split


class HighD(Dataset):
    def __init__(self, raw_data_dir, processed_dir, obs_len, pred_len, process_data = False, heading_threshold = 0.01) -> None:
        super().__init__()
        self.raw_data_dir = raw_data_dir
        self.processed_dir = processed_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.heading_threshold = heading_threshold
        self.scene_data_list = []
        if process_data:
            for i in tqdm(range(1, 61, 1), desc='Processing data'):
                self.scene_data_list.extend(self.generate_training_data(i))
        else:
            self.scene_data_list = torch.load(self.processed_dir + 'scene_list.pt')
    def __getitem__(self, id):
        return self.scene_data_list[id]
    
    def __len__(self):
        return len(self.scene_data_list)
    
    
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
            if end_frame_idx >= lane_change_info[i][0] and end_frame_idx <= lane_change_info[i][2]:
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
        i = start_frame_idx
        while (i < start_frame_idx + self.obs_len):
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
        
        surrounding_feature.append(preceding_feature)
        surrounding_feature.append(following_feature)
        surrounding_feature.append(left_following_feature)
        surrounding_feature.append(left_alongside_feature)
        surrounding_feature.append(left_preceding_feature)
        surrounding_feature.append(right_following_feature)
        surrounding_feature.append(right_alongside_feature)
        surrounding_feature.append(right_preceding_feature)
        
        return target_feature, surrounding_feature
            
        
            
            
    
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
        
        #2. 获取变道轨迹的特征
        for id in tqdm(lane_changing_ids, desc="generate lane changing scene"):
            lane_change_info = self.get_lane_changing_info(tracks_csv[id], lane_num)
            driving_direction = tracks_meta[id][DRIVING_DIRECTION]
            for i in range(len(tracks_csv[id][FRAME]) - self.obs_len + 1):
                target_feature, surroungding_feature = self.construct_traj_features(tracks_csv, tracks_meta, id, i, lanes_info, driving_direction)
                lane_change_label = self.get_traj_label(i + self.obs_len - 1, lane_change_info)
                scene_dict = {}
                scene_dict["target_obs_traj"] = torch.tensor(target_feature) #(obs_len, feature_dim)
                scene_dict["surrounding_obs_traj"] = torch.tensor(surroungding_feature) #(8, obs_len, feature_dim)
                scene_dict["lane_change_label"] = torch.tensor(lane_change_label) #(1, )
                scene_list.append(scene_dict)
                if target_feature[-1][1] < -2 and lane_change_label == 1:
                    print(lane_change_info, id, i)
        print(len(scene_list))
        #3. 获取直行轨迹的特征
        # for id in tqdm(lane_keeping_ids, desc="generate lane keeping scene"):
        #     driving_direction = tracks_meta[id][DRIVING_DIRECTION]
        #     for i in range(len(tracks_csv[id][FRAME]) - self.obs_len + 1):
        #         target_feature, surroungding_feature = self.construct_traj_features(tracks_csv, tracks_meta, id, i, lanes_info, driving_direction)
        #         lane_change_label = 0.
        #         scene_dict = {}
        #         scene_dict["target_obs_traj"] = torch.tensor(target_feature) #(obs_len, feature_dim)
        #         scene_dict["surrounding_obs_traj"] = torch.tensor(surroungding_feature) #(8, obs_len, feature_dim)
        #         scene_dict["lane_change_label"] = torch.tensor(lane_change_label) #(1, )
        #         scene_list.append(scene_dict)
        print(len(scene_list))
        return scene_list
    
def get_label_weight(scene_list):
    label_num = [0, 0, 0]
    for scene in scene_list:
        if scene['lane_change_label'] == 0:
            label_num[0] += 1
        elif scene['lane_change_label'] == 1:
            label_num[1] += 1
        else:
            label_num[2] += 1
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