from dataset.highD.data_processing import HighD
from dataset.highD.utils import *
import torch
from matplotlib import pyplot as plt

# dataset = HighD(RAW_DATA_DIR, 30, 20, 0.03)
# scene_list = dataset.generate_training_data(1)
# scene_list = torch.save(scene_list, 'scene_list.pt')

scene_list = torch.load('scene_list.pt')
for scene in scene_list:
    target_traj = scene['target_obs_traj']
    label = scene['lane_change_label']
    surounding_traj = scene['surrounding_obs_traj']
    plt.plot(target_traj[:, 0], target_traj[:, 1])
    for i in range(surounding_traj.shape[0]):
        plt.plot(surounding_traj[i, :, 0], surounding_traj[i, :, 1])
    plt.show()