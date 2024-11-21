'''
Author: Yang Jialong
Date: 2024-11-11 17:33:57
LastEditTime: 2024-11-20 16:56:25
Description: 请填写简介
'''
from dataset.highD.data_processing import HighD
from dataset.highD.utils import *
from modules.trajectory_generator import TrajectoryGenerator
import torch
from matplotlib import pyplot as plt

dataset = HighD(RAW_DATA_DIR, PROCESSED_DATA_DIR, 50, 75)
scene_list = dataset.generate_training_data(1)
# scene_list = torch.save(scene_list, 'scene_list.pt')

# scene_list = torch.load('scene_list.pt')
# for scene in scene_list:
#     target_traj = scene['target_obs_traj']
#     label = scene['lane_change_label']
#     surounding_traj = scene['surrounding_obs_traj']
#     plt.plot(target_traj[:, 0], target_traj[:, 1])
#     for i in range(surounding_traj.shape[0]):
#         plt.plot(surounding_traj[i, :, 0], surounding_traj[i, :, 1])
#     plt.show()
