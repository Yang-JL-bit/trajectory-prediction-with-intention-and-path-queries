'''
Author: Yang Jialong
Date: 2024-12-02 10:34:02
LastEditTime: 2025-01-11 15:56:21
Description: 请填写简介
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

def loss_fn_traj(
    intention_cls_pred,  # (bs, 3)
    intention_cls_label,  # (bs,)
    traj_score_pred,  # (bs, n_candidate)
    traj_gt,  # (bs, n_pred, 2)
    candidate_traj,  # (bs, n_candidate, n_pred, 2)
    candidate_traj_mask,  # (bs, n_candidate)
    device = 'CPU',
    alpha=1.0,  # 权重系数
    beta=1.0   # 权重系数
):
    # 1. 意图分类损失 (交叉熵)
    criterion_cls = nn.CrossEntropyLoss().to(device)
    loss_intention_cls = criterion_cls(intention_cls_pred, intention_cls_label)

    # 2. 轨迹分类损失
    batch_size, n_candidate, n_pred, _ = candidate_traj.shape

    # 计算每个候选轨迹和真实轨迹的距离 (bs, n_candidate)
    traj_distances = torch.norm(
        candidate_traj - traj_gt.unsqueeze(1), dim=-1
    ).sum(dim=-1)  # 路径上所有点的 L2 距离

    # 找到最接近真实轨迹的轨迹索引
    closest_traj_idx = torch.argmin(traj_distances + (1 - candidate_traj_mask) * 1e6, dim=1)

    # 生成轨迹分类标签 (bs, n_candidate)
    traj_cls_label = torch.zeros_like(traj_score_pred, dtype=torch.long)  # 全部初始化为 0
    traj_cls_label.scatter_(1, closest_traj_idx.unsqueeze(1), 1)  # 设置最近轨迹的标签为 1
    traj_score_pred = traj_score_pred + (candidate_traj_mask - 1)  * 1e9

    loss_traj_cls = F.cross_entropy(traj_score_pred, traj_cls_label.argmax(dim=1))


    # 3. 轨迹预测误差 (Smooth L1)
    # 获取最接近真实轨迹的候选轨迹 (bs, n_pred, 2)
    traj_score_pred_idx = traj_score_pred.argmax(dim=1)
    closest_candidate_traj = torch.gather(
        candidate_traj,
        1,
        traj_score_pred_idx.view(batch_size, 1, 1, 1).expand(-1, -1, n_pred, 2)
    ).squeeze(1)  # 去掉轨迹选择的维度

    # Smooth L1 损失    
    # loss_traj_reg = F.smooth_l1_loss(closest_candidate_traj, traj_gt, reduction="mean")
        
    criterion_reg = nn.SmoothL1Loss().to(device)
    loss_traj_reg = criterion_reg(closest_candidate_traj, traj_gt)
    
    #test plot
    # for traj in candidate_traj[0]:
    #     plt.plot(traj[:,0],traj[:,1],'g-')
    # plt.plot(closest_candidate_traj[0,:,0],closest_candidate_traj[0,:,1],'r-')
    # plt.plot(traj_gt[0,:,0],traj_gt[0,:,1],'b-')
    # plt.show()
    

    # 4. 总损失
    total_loss = loss_intention_cls + alpha * loss_traj_cls + beta * loss_traj_reg
    return total_loss, loss_intention_cls, loss_traj_cls, loss_traj_reg

