import torch
from sklearn.cluster import KMeans

def cluster_trajectories(dataset, num_clusters):
    # 提取直行、左转和右转的轨迹
    straight_trajs = torch.stack([data['future_traj_gt'] for data in dataset if data['lane_change_label'] == 0])
    left_turn_trajs = torch.stack([data['future_traj_gt'] for data in dataset if data['lane_change_label'] == 1])
    right_turn_trajs = torch.stack([data['future_traj_gt'] for data in dataset if data['lane_change_label'] == 2])

    # 定义一个函数来对轨迹进行聚类
    def cluster_trajs(trajs, num_clusters):
        if trajs.size(0) == 0:  # 如果没有轨迹，返回空列表
            return []

        # 计算轨迹之间的距离矩阵
        # trajs的形状: (num_trajs, time_steps, 2)
        # 使用欧几里得距离计算每对轨迹之间的距离
        trajs_expanded = trajs.unsqueeze(1)  # (num_trajs, 1, time_steps, 2)
        trajs_expanded_t = trajs.unsqueeze(0)  # (1, num_trajs, time_steps, 2)
        distance_matrix = torch.norm(trajs_expanded - trajs_expanded_t, dim=3).sum(dim=2)  # (num_trajs, num_trajs)

        # 使用KMeans进行聚类
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        labels = kmeans.fit_predict(distance_matrix.cpu().numpy())

        # 计算每个聚类的中心轨迹
        cluster_centers = []
        for cluster_id in range(num_clusters):
            cluster_mask = torch.tensor(labels) == cluster_id
            if cluster_mask.any():
                cluster_center = trajs[cluster_mask].mean(dim=0)  # 计算平均轨迹
                cluster_centers.append(cluster_center)

        return cluster_centers

    # 对直行、左转和右转的轨迹分别进行聚类
    straight_centers = cluster_trajs(straight_trajs, num_clusters)
    left_turn_centers = cluster_trajs(left_turn_trajs, num_clusters)
    right_turn_centers = cluster_trajs(right_turn_trajs, num_clusters)

    return {
        'straight': straight_centers,
        'left_turn': left_turn_centers,
        'right_turn': right_turn_centers
    }


def cluster_last_points_by_label(dataset, num_clusters):
    # 提取直行、左转和右转的轨迹的最后一个点
    straight_last_points = torch.stack([data['future_traj_gt'][-1] for data in dataset if data['lane_change_label'] == 0])
    left_turn_last_points = torch.stack([data['future_traj_gt'][-1] for data in dataset if data['lane_change_label'] == 1])
    right_turn_last_points = torch.stack([data['future_traj_gt'][-1] for data in dataset if data['lane_change_label'] == 2])

    # 定义一个函数来对点进行聚类
    def cluster_points(points, num_clusters):
        if points.size(0) == 0:  # 如果没有点，返回空列表
            return []

        # 使用KMeans进行聚类
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        labels = kmeans.fit_predict(points.cpu().numpy())

        # 计算每个聚类的中心点
        cluster_centers = []
        for cluster_id in range(num_clusters):
            cluster_mask = torch.tensor(labels) == cluster_id
            if cluster_mask.any():
                cluster_center = points[cluster_mask].mean(dim=0)  # 计算平均点
                cluster_centers.append(cluster_center)

        return cluster_centers

    # 对直行、左转和右转的最后一个点分别进行聚类
    straight_centers = cluster_points(straight_last_points, num_clusters)
    left_turn_centers = cluster_points(left_turn_last_points, num_clusters)
    right_turn_centers = cluster_points(right_turn_last_points, num_clusters)

    return {
        'straight': straight_centers,
        'left_turn': left_turn_centers,
        'right_turn': right_turn_centers
    }


import matplotlib.pyplot as plt

def plot_cluster_centers(cluster_centers):
    """
    绘制聚类中心的轨迹
    :param cluster_centers: cluster_trajectories 函数的输出，包含直行、左转和右转的聚类中心
    """
    # 提取直行、左转和右转的聚类中心
    straight_centers = cluster_centers['straight']
    left_turn_centers = cluster_centers['left_turn']
    right_turn_centers = cluster_centers['right_turn']

    # 创建一个画布
    plt.figure(figsize=(10, 6))

    # 绘制直行的聚类中心
    for i, center in enumerate(straight_centers):
        center = center.cpu().numpy()  # 将轨迹从 GPU 移动到 CPU（如果必要）
        plt.plot(center[:, 0], center[:, 1], label=f'Straight Cluster {i+1}', linestyle='-', marker='o')

    # 绘制左转的聚类中心
    for i, center in enumerate(left_turn_centers):
        center = center.cpu().numpy()  # 将轨迹从 GPU 移动到 CPU（如果必要）
        plt.plot(center[:, 0], center[:, 1], label=f'Left Turn Cluster {i+1}', linestyle='--', marker='s')

    # 绘制右转的聚类中心
    for i, center in enumerate(right_turn_centers):
        center = center.cpu().numpy()  # 将轨迹从 GPU 移动到 CPU（如果必要）
        plt.plot(center[:, 0], center[:, 1], label=f'Right Turn Cluster {i+1}', linestyle=':', marker='^')

    # 添加图例和标签
    plt.title('Cluster Centers of Trajectories')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

# 示例调用

def plot_endpoint_cluster_centers(cluster_centers):
    # 提取每个方向的聚类中心
    straight_centers = cluster_centers['straight']
    left_turn_centers = cluster_centers['left_turn']
    right_turn_centers = cluster_centers['right_turn']

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制直行的聚类中心
    if straight_centers:
        straight_centers = torch.stack(straight_centers).cpu().numpy()
        plt.scatter(straight_centers[:, 0], straight_centers[:, 1], c='blue', label='Straight', marker='o')

    # 绘制左转的聚类中心
    if left_turn_centers:
        left_turn_centers = torch.stack(left_turn_centers).cpu().numpy()
        plt.scatter(left_turn_centers[:, 0], left_turn_centers[:, 1], c='green', label='Left Turn', marker='^')

    # 绘制右转的聚类中心
    if right_turn_centers:
        right_turn_centers = torch.stack(right_turn_centers).cpu().numpy()
        plt.scatter(right_turn_centers[:, 0], right_turn_centers[:, 1], c='red', label='Right Turn', marker='s')

    # 添加图例和标签
    plt.title('Cluster Centers of Trajectory Last Points')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()