'''
Author: Yang Jialong
Date: 2024-12-02 10:34:02
LastEditTime: 2025-02-26 16:06:21
Description: 损失函数
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

def compute_traj_loss(combined_confidence_score, 
                            intention_label,
                            traj_gt,
                            traj_prior):
    """
    计算优选轨迹置信度的交叉熵损失
    
    参数：
    combined_confidence_score : torch.Tensor (bs, 18)
        候选轨迹的置信度分数
    intention_label : torch.Tensor (bs, 3)
        意图类别概率
    traj_gt : torch.Tensor (bs, n_predic, 2)
        真实轨迹坐标
    traj_prior : torch.Tensor (bs, 3, 6, n_predic, 2)
        先验轨迹库
    
    返回：
    loss : torch.Tensor
        交叉熵损失值
    """
    bs = combined_confidence_score.size(0)
    device = intention_label.device
    # Step 1. 获取主导意图
    intention_idx = intention_label
    
    # Step 2. 提取对应意图的先验轨迹
    # traj_prior[bs, 3, 6, T, 2] -> selected_prior[bs, 6, T, 2]
    traj_prior = traj_prior.to(device)
    selected_prior = traj_prior[torch.arange(bs), intention_idx]
    
    # Step 3. 计算欧式距离（平方和）
    # 扩展GT维度用于广播计算
    gt_expanded = traj_gt.unsqueeze(1)  # (bs, 1, T, 2)
    
    # 计算所有候选轨迹与GT的差异
    diff = selected_prior - gt_expanded  # (bs, 6, T, 2)
    
    # 计算每个候选轨迹的平方距离和（按时间和坐标维度）
    distance = (diff ** 2).sum(dim=(-1, -2))  # (bs, 6)
    
    # Step 4. 找到最近邻轨迹的索引
    closest_idx = distance.argmin(dim=1)  # (bs,)
    
    # Step 5. 转换为全局索引（intention_id*6 + mode_id）
    global_indices = intention_idx * 6 + closest_idx  # (bs,)
    
    # Step 6. 计算交叉熵损失
    loss_fn = nn.CrossEntropyLoss().to(device)
    loss = loss_fn(combined_confidence_score, global_indices)
    
    return loss

def convert_prior_dict_to_tensor(prior_dict, batch_size, n_predic):
    """
    将先验轨迹字典转换为张量格式
    
    参数：
    prior_dict : dict
        先验轨迹字典，键为意图类别，值为包含6个(n_predic, 2)张量的列表
    batch_size : int
        批次大小
    n_predic : int
        预测时间步数
    
    返回：
    prior_tensor : torch.Tensor (bs, 3, 6, n_predic, 2)
        转换后的先验轨迹张量
    """
    # 定义意图顺序（与intention_label的维度顺序一致）
    intent_order = ["straight", "left_turn", "right_turn"]
    
    # 初始化结果张量
    prior_tensor = torch.zeros(batch_size, 3, 6, n_predic, 2)
    
    # 遍历每个意图
    for intent_idx, intent_key in enumerate(intent_order):
        # 获取当前意图的6个候选轨迹
        traj_list = prior_dict[intent_key]  # 6个(n_predic, 2)的张量
        
        # 将列表中的轨迹堆叠成张量
        traj_tensor = torch.stack(traj_list)  # (6, n_predic, 2)
        
        # 将当前意图的轨迹复制到整个批次
        prior_tensor[:, intent_idx] = traj_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1)
    return prior_tensor


def loss_fn_traj(
    intention_cls_pred,  # (bs, 3)
    intention_cls_label,  # (bs,)
    traj_score_pred,  # (bs, n_candidate)
    traj_gt,  # (bs, n_pred, 2)
    candidate_traj,  # (bs, n_candidate, n_pred, 2)
    candidate_traj_mask,  # (bs, n_candidate)
    combined_confidence_score,  # (bs, 18)
    traj_prior_dict,  # (bs, 3, 6, n_pred, 2)
    endpoint,  # (bs, n_pred, 2)
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
    )[:,:, -1]  # 路径上所有点的 L2 距离

    # 找到最接近真实轨迹的轨迹索引
    closest_traj_idx = torch.argmin(traj_distances + (1 - candidate_traj_mask) * 1e6, dim=1)

    # 生成轨迹分类标签 (bs, n_candidate)
    # traj_cls_label = torch.zeros_like(traj_score_pred, dtype=torch.long)  # 全部初始化为 0
    # traj_cls_label.scatter_(1, closest_traj_idx.unsqueeze(1), 1)  # 设置最近轨迹的标签为 1
    # traj_score_pred = traj_score_pred + (candidate_traj_mask - 1)  * 1e9
    # loss_traj_cls = F.cross_entropy(traj_score_pred, traj_cls_label.argmax(dim=1))
    
    traj_prior = convert_prior_dict_to_tensor(traj_prior_dict, batch_size, n_pred)
    loss_traj_cls = compute_traj_loss(combined_confidence_score, intention_cls_label, traj_gt, traj_prior)
    # 3. 轨迹预测误差 (Smooth L1)
    # 获取最接近真实轨迹的候选轨迹 (bs, n_pred, 2)
    traj_score_pred_idx = traj_score_pred.argmax(dim=1)
    closest_candidate_traj = torch.gather(
        candidate_traj,
        1,
        traj_score_pred_idx.view(batch_size, 1, 1, 1).expand(-1, -1, n_pred, 2)
    ).squeeze(1)  # 去掉轨迹选择的维度
        
    criterion_reg = nn.SmoothL1Loss().to(device)
    loss_traj_reg = criterion_reg(closest_candidate_traj, traj_gt)
    
    # 加一个endpoint loss
    closet_endpoint = torch.gather(
        endpoint,
        1,
        traj_score_pred_idx.view(batch_size, 1, 1).expand(-1, -1, 2)
    ).squeeze(1)
    endpoint_loss = F.smooth_l1_loss(closet_endpoint, traj_gt[:, -1, :], reduction="mean")
    # 4. 总损失（去掉了endpointloss）
    total_loss = loss_intention_cls + alpha * loss_traj_cls + beta * loss_traj_reg
    
    return total_loss, loss_intention_cls, loss_traj_cls, loss_traj_reg, endpoint_loss

