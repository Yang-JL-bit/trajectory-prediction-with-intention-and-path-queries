'''
Author: Yang Jialong
Date: 2024-12-05 09:11:11
LastEditTime: 2024-12-25 16:47:56
Description: 可视化
'''

import torch
import numpy as np
import matplotlib.pyplot as plt

def visualization(traj_score_pred, candidate_trajectory, candidate_traj_mask, future_trajectory_gt, top_k = 6):
    # 对无效轨迹分数赋值为 -inf
    valid_scores = traj_score_pred + (1 - candidate_traj_mask) * (-1e9)  # (bs, n_candidate)

    # 选取分数最高的 top_k 有效轨迹
    top_k_scores, top_k_indices = torch.topk(valid_scores, top_k, dim=-1)  # (bs, top_k)
    top_k_traj = torch.gather(candidate_trajectory, 1, top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, candidate_trajectory.size(2), 2))  # (bs, top_k, n_pred, 2)
    
    #画出topk轨迹和轨迹真值
    for j in range(top_k_traj.shape[0]):
        for i in range(top_k):
            plt.plot(top_k_traj[j,i,:,0].cpu(),top_k_traj[j,i,:,1].cpu(),'blue')
        plt.plot(future_trajectory_gt[j,:,0].cpu(),future_trajectory_gt[j,:,1].cpu(),'red')
        plt.savefig(f'./fig/{j}.png')
        plt.close()


def plot_train_loss(save_path):
    train_loss_list = torch.load(save_path)
    train_loss = [loss["traj_reg_loss"] for loss in train_loss_list]
    plt.plot(train_loss)
    plt.show()

if __name__ == '__main__':
    plot_train_loss('./save/1225_training_2/train_loss_list.pth')
