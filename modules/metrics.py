'''
Author: Yang Jialong
Date: 2024-12-03 14:47:23
LastEditTime: 2025-01-16 15:51:12
Description: 请填写简介
'''
import torch
import numpy as np

def cal_intention_acc(pred, label):
    pred = torch.argmax(pred, dim=-1)
    acc = torch.sum(pred == label) / len(pred)
    return acc


def cal_traj_acc(traj_score_pred, candidate_traj,  traj_gt, candidate_traj_mask):
    traj_distances = torch.norm(candidate_traj - traj_gt.unsqueeze(1), dim=-1).sum(dim=-1)  # 路径上所有点的 L2 距离
    # 找到最接近真实轨迹的轨迹索引
    closest_traj_idx = torch.argmin(traj_distances + (1 - candidate_traj_mask) * 1e6, dim=1)
    traj_score_pred = traj_score_pred + (candidate_traj_mask - 1)  * 1e9
    traj_score_pred_idx = torch.argmax(traj_score_pred, dim=-1)
    # print("label: ", closest_traj_idx)
    # print("pred: ", traj_score_pred_idx)
    acc = torch.sum(traj_score_pred_idx == closest_traj_idx) / len(closest_traj_idx)
    return acc


def cal_minADE(traj_score_pred, candidate_trajectory, candidate_traj_mask, traj_gt, top_k=6):
    """
    计算 minADE (最小平均距离误差)，仅基于有效轨迹中的分数最高的 top_k 条轨迹
    
    参数:
    - traj_score_pred: Tensor, (bs, n_candidate), 候选轨迹分数预测
    - candidate_trajectory: Tensor, (bs, n_candidate, n_pred, 2), 候选轨迹
    - candidate_traj_mask: Tensor, (bs, n_candidate), 候选轨迹掩码
    - traj_gt: Tensor, (bs, n_pred, 2), 真实轨迹
    - top_k: int, 计算范围内的候选轨迹数量
    
    返回:
    - minADE: Tensor, (bs,), 每个样本的 minADE
    """
    # # 对无效轨迹分数赋值为 -inf
    # valid_scores = traj_score_pred + (1 - candidate_traj_mask) * (-1e9)  # (bs, n_candidate)

    # # 选取分数最高的 top_k 有效轨迹
    # top_k_scores, top_k_indices = torch.topk(valid_scores, top_k, dim=-1)  # (bs, top_k)
    # top_k_traj = torch.gather(candidate_trajectory, 1, top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, candidate_trajectory.size(2), 2))  # (bs, top_k, n_pred, 2)
    top_k_traj = candidate_trajectory
    # 计算每个候选轨迹与真实轨迹的平均欧几里得距离
    traj_gt_expanded = traj_gt.unsqueeze(1).expand(-1, top_k, -1, -1)  # (bs, top_k, n_pred, 2)
    distances = torch.norm(top_k_traj - traj_gt_expanded, dim=-1)  # (bs, top_k, n_pred)
    avg_distances = distances.mean(dim=-1)  # (bs, top_k)

    # 计算最小平均距离
    minADE = avg_distances.min(dim=-1)[0]  # (bs,)
    
    return minADE


def cal_minFDE(traj_score_pred, candidate_trajectory, candidate_traj_mask, traj_gt, top_k=6):
    """
    计算 minFDE (最小终点距离误差)，仅基于有效轨迹中的分数最高的 top_k 条轨迹
    
    参数:
    - traj_score_pred: Tensor, (bs, n_candidate), 候选轨迹分数预测
    - candidate_trajectory: Tensor, (bs, n_candidate, n_pred, 2), 候选轨迹
    - candidate_traj_mask: Tensor, (bs, n_candidate), 候选轨迹掩码
    - traj_gt: Tensor, (bs, n_pred, 2), 真实轨迹
    - top_k: int, 计算范围内的候选轨迹数量
    
    返回:
    - minFDE: Tensor, (bs,), 每个样本的 minFDE
    """
    # # 对无效轨迹分数赋值为 -inf
    # valid_scores = traj_score_pred + (1 - candidate_traj_mask) * (-1e9)  # (bs, n_candidate)

    # # 选取分数最高的 top_k 有效轨迹
    # top_k_scores, top_k_indices = torch.topk(valid_scores, top_k, dim=-1)  # (bs, top_k)
    # top_k_traj = torch.gather(candidate_trajectory, 1, top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, candidate_trajectory.size(2), 2))  # (bs, top_k, n_pred, 2)
    top_k_traj = candidate_trajectory
    # 计算每个候选轨迹的终点与真实轨迹终点的欧几里得距离
    traj_gt_end = traj_gt[:, -1, :]  # (bs, 2)
    top_k_traj_end = top_k_traj[:, :, -1, :]  # (bs, top_k, 2)
    distances = torch.norm(top_k_traj_end - traj_gt_end.unsqueeze(1), dim=-1)  # (bs, top_k)

    # 计算最小终点距离
    minFDE = distances.min(dim=-1)[0]  # (bs,)
    return minFDE


def cal_miss_rate(traj_score_pred, candidate_trajectory, candidate_traj_mask, traj_gt, top_k=6, threshold=2.0):
    """
    计算 Miss Rate，在预测的最后一个时间步，top-k 预测轨迹与真实轨迹的距离均大于 2m 的比例

    参数:
    - traj_score_pred: Tensor, (bs, n_candidate), 候选轨迹分数预测
    - candidate_trajectory: Tensor, (bs, n_candidate, n_pred, 2), 候选轨迹
    - candidate_traj_mask: Tensor, (bs, n_candidate), 候选轨迹掩码
    - traj_gt: Tensor, (bs, n_pred, 2), 真实轨迹
    - top_k: int, 使用的候选轨迹数量
    - threshold: float, Miss 判定的距离阈值 (单位: 米)

    返回:
    - miss_rate: float, Miss Rate 的比例
    """
    # # 对无效轨迹分数赋值为 -inf
    # valid_scores = traj_score_pred + (1 - candidate_traj_mask) * (-1e9)  # (bs, n_candidate)

    # # 选取分数最高的 top-k 有效轨迹
    # top_k_scores, top_k_indices = torch.topk(valid_scores, top_k, dim=-1)  # (bs, top_k)
    # top_k_traj = torch.gather(candidate_trajectory, 1, 
    #                           top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, candidate_trajectory.size(2), 2))  # (bs, top_k, n_pred, 2)
    top_k_traj = candidate_trajectory
    # 获取 top-k 轨迹的终点
    top_k_traj_end = top_k_traj[:, :, -1, :]  # (bs, top_k, 2)

    # 获取真实轨迹的终点
    traj_gt_end = traj_gt[:, -1, :]  # (bs, 2)

    # 计算每个 top-k 轨迹终点与真实轨迹终点的欧几里得距离
    distances = torch.norm(top_k_traj_end - traj_gt_end.unsqueeze(1), dim=-1)  # (bs, top_k)

    # 判断是否 Miss
    miss_mask = (distances > threshold).all(dim=-1).float()  # (bs,), 每个样本是否 Miss

    # 计算 Miss Rate
    miss_rate = miss_mask.mean().item()  # scalar, Miss Rate 的比例
    return miss_rate

def cal_offroad_rate(future_pred, lanes_info):
    bs, n_candidate, n_pred, _ = future_pred.shape
    offroad_mask = torch.zeros((bs, n_candidate), dtype=torch.float32)
    for i in range(bs):
        lane_info = lanes_info[i]
        for j in range(n_candidate):
            for k in range(n_pred):
                y_coord = future_pred[i, j, k, 1].item()
                valid = False
                for lane_id, lane_boundaries in lane_info.items():
                    left_boundary, center_lane, right_boundary = lane_boundaries
                    if left_boundary <= y_coord <= right_boundary:
                        valid = True
                        break
                if not valid:
                    offroad_mask[i, j] = 1.0
                    break

    offroad_rate = torch.mean(offroad_mask)
    return offroad_rate


def cal_kinematic_feasibility_rate(future_pred, max_curvature=0.2, delta_x = 0.2):
    bs, n_candidate, n_pred, _ = future_pred.shape
    infeasible_mask = torch.zeros((bs, n_candidate), dtype=torch.float32)
    
    # 转换为numpy处理
    trajectories = future_pred.detach().cpu().numpy()
    
    for i in range(bs):
        for j in range(n_candidate):
            traj = trajectories[i, j]  # (n_pred, 2)
            
            # 计算一阶导数
            dx = np.gradient(traj[:, 0], 0.2)
            dy = np.gradient(traj[:, 1], 0.2)
            
            # 计算二阶导数
            d2x = np.gradient(dx, 0.2)
            d2y = np.gradient(dy, 0.2)
            print("dx: {}".format(dx))
            print("dy: {}".format(dy))
            # 计算曲率
            denominator = (dx**2 + dy**2)**1.5
            curvature = np.abs(np.where(denominator > 1e-6,
                                      (dx * d2y - d2x * dy) / denominator,
                                      0))
            # 判断是否超过最大曲率
            if np.any(curvature > max_curvature):
                infeasible_mask[i, j] = 1.0
    
    # 计算不符合运动学约束的比例
    infeasible_rate = torch.mean(infeasible_mask)
    return infeasible_rate