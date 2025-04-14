import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def square_distance(src, dst):
    """
    计算两组点之间的成对距离平方
    src: B, N, C
    dst: B, M, C
    返回: B, N, M 的距离矩阵
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    # 假设在第1个点云文件中：
    # src中的第1个中心点: [1,2,3]
    # dst中的某个点: [4,5,6]
    # 距离计算过程：
    # 1. - 2 < src, dst > = -2 * (1 * 4 + 2 * 5 + 3 * 6)
    # 2. | | src | |² = 1² + 2² + 3²
    # 3. | | dst | |² = 4² + 5² + 6²
    # 4. 最终距离 = | | src | |² + | | dst | |² - 2 < src, dst >

    # 1. 计算内积项：-2<src,dst>
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # src: [48, 256, 3]
    # dst.permute(0,2,1): [48, 3, 4096]
    # 结果: [48, 256, 4096]

    # 2. 添加src的平方和项
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    # torch.sum(src ** 2, -1): [48, 256]
    # .view(B, N, 1): [48, 256, 1]
    # 广播到: [48, 256, 4096]

    # 3. 添加dst的平方和项
    dist += torch.sum(dst ** 2, -1).view(B, 1, M) #dst ** 2：对每个元素平方；torch.sum(..., -1)：在最后一个维度上求和。
    # torch.sum(dst ** 2, -1): [48, 4096]
    # .view(B, 1, M): [48, 1, 4096]
    # 广播到: [48, 256, 4096]

    return dist


def index_points(points, idx):
    """
    points: [B, N, C]
    idx: [B, S]
    return: [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    最远点采样
    xyz: B, N, 3
    npoint: 需要采样的点数
    返回: B, npoint 的采样点索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)#存储采样点的索引
    distance = torch.ones(B, N).to(device) * 1e10 # 初始化距离矩阵，设置为一个很大的值(1e10)
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)# 随机选择初始点（每个批次随机选择一个点）；0是随机数下界，N是随机数上界，（B，）是输出张量的形状，dtype=torch.long是指定数据类型为长整型
    batch_indices = torch.arange(B, dtype=torch.long).to(device)# 创建批次索引，用于批处理操作；

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)#(xyz - centroid)：广播减法；torch.sum(..., -1)：在最后一个维度上求和
        distance = torch.min(distance, dist) # distance 始终记录的是 每个未选点 到 当前已选点集 的最近距离。
        farthest = torch.max(distance, -1)[1]# 从中找到“最近距离”最大的点，作为下一个最远点。
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    球形邻域查询
    radius: 球体半径
    nsample: 每个球体中最多采样的点数
    xyz:全部点坐标 B, N, 3
    new_xyz:中心点坐标 B, S, 3
    返回: B, S, nsample 的采样点索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N # 将超出半径的点的索引设置为N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]# 对每个球体内的点进行排序并只保留前nsample个

    # 处理球体内点数不足的情况
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])# 获取每个球体中第一个点的索引，复制到需要的大小
    mask = group_idx == N # 创建掩码标识无效点（索引为N的点）
    group_idx[mask] = group_first[mask]
    return group_idx


class PointNetSetAbstractionMsg(nn.Module): #Msg = Multi-Scale Grouping
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint# 设置采样点数
        self.radius_list = radius_list# 设置多尺度球形邻域的半径列表    e.g. [0.1, 0.2, 0.4]
        self.nsample_list = nsample_list# 设置每个尺度下采样的邻居点数  e.g. [ 16,  32,  64]
        self.mlp_convs = nn.ModuleList()# 创建存储卷积层的ModuleList
        self.mlp_bns = nn.ModuleList()# 创建存储批归一化层的ModuleList

        for i in range(len(mlp_list)):# 遍历每个尺度的MLP配置
            convs = nn.ModuleList()# 当前尺度的卷积层列表
            bns = nn.ModuleList()# 当前尺度的批归一化层列表
            last_channel = in_channel + 3 # 初始输入通道数(特征维度 + xyz坐标)
            for out_channel in mlp_list[i]:# 构建当前尺度的MLP网络
                convs.append(nn.Conv2d(last_channel, out_channel, 1))# 添加1x1卷积层
                bns.append(nn.BatchNorm2d(out_channel))# 添加批归一化层
                last_channel = out_channel# 更新输入通道数
            self.mlp_convs.append(convs)# 将当前尺度的卷积层添加到总列表
            self.mlp_bns.append(bns)# 将当前尺度的批归一化层添加到总列表

    def forward(self, xyz, points):
        """
        xyz: B, N, 3
        points: B, N, C
        """
        # 移除之前的转置操作，因为输入已经是正确的形状
        B, N, C = xyz.shape
        S = self.npoint

        # 直接使用 xyz 进行采样
        centroids = farthest_point_sample(xyz, S)  # [B, S] centroids是指将最远点当作中心点
        new_xyz = index_points(xyz, centroids)  # [B, S, 3] 获取采样点centroids的坐标

        new_points_list = []

        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]# 获取当前尺度的邻居点数
            group_idx = query_ball_point(radius, K, xyz, new_xyz)# 查询球形邻域内的点的索引
            grouped_xyz = index_points(xyz, group_idx) # 获取组内点的坐标 # [B, S, K, 3]
            grouped_xyz -= new_xyz.view(B, S, 1, 3)# 计算组内点相对采样点centroids的坐标

            if points is not None:
                grouped_points = index_points(points, group_idx) # 获取组内点的特征 # [B, S, K, C]
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1) #组内点的坐标与特征拼接 # [B, S, K, C+3]
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, C+3, K, S]

            for j in range(len(self.mlp_convs[i])):
                conv = self.mlp_convs[i][j]
                bn = self.mlp_bns[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))

            new_points = torch.max(grouped_points, 2)[0] # dim=2 表示在K这个维度上取最大值；[0] 表示只取最大值，不要对应的索引；  # [B, D', S]
            new_points_list.append(new_points)

        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        xyz: B, N, 3
        points: B, C, N 或 B, N, C
        """
        # 确保 points 的维度正确
        if points is not None and points.shape[1] != xyz.shape[1]:
            points = points.transpose(1, 2)  # [B, N, C]

        if self.group_all:
            new_xyz = torch.zeros(xyz.shape[0], 1, 3).to(xyz.device)
            if points is not None:
                new_points = torch.cat([xyz, points], dim=2)  # [B, N, C+3]
            else:
                new_points = xyz
            new_points = new_points.transpose(1, 2).unsqueeze(-1)  # [B, C+3, N, 1]
            grouped_points = new_points
        else:
            new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))
            group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)  # [B, npoint, nsample, 3]
            grouped_xyz -= new_xyz.view(xyz.shape[0], self.npoint, 1, 3)

            if points is not None:
                grouped_points = index_points(points, group_idx)  # [B, npoint, nsample, C]
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # [B, npoint, nsample, C+3]
            else:
                grouped_points = grouped_xyz  # [B, npoint, nsample, 3]

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, C+3, nsample, npoint]

        # 应用 MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))

        # 最大池化
        new_points = torch.max(grouped_points, 2)[0]  # [B, C, npoint] 或 [B, C, 1]

        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: B, N, 3 需要插值的目标点坐标（高分辨率）
        xyz2: B, S, 3 已知特征点的坐标（低分辨率）
        points1: B, N, C (可选)目标点的特征
        points2: B, S, C 已知点的特征
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)# 计算距离矩阵
            dists, idx = dists.sort(dim=-1)# 对距离排序，获取最近的3个点
            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)# 计算距离的倒数作为初始权重
            norm = torch.sum(dist_recip, dim=2, keepdim=True)# 权重归一化
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points