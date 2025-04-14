from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math
from models.pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(
            feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(
            1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        # 修改特征传播层的输入维度
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=256,   # 采样后的点云数量，从原始点云中采样256个点
            radius_list=[0.1, 0.2, 0.4], # 三个不同尺度的球形邻域半径
            nsample_list=[16, 32, 64], # 每个尺度下采样的邻居点数量
            in_channel=1024, # 输入特征维度
            mlp_list=[[16, 16, 32], [32, 32, 64], [32, 64, 64]]# 每个尺度的MLP层配置
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=160,
            mlp=[256, 512],
            group_all=True
        )
        # 修改特征传播层的输入维度
        # fp2的输入维度应该是 l1_points.shape[1] + l2_points.shape[1]
        self.fp2 = PointNetFeaturePropagation(160 + 512, [256, 128])  # 160 from sa1, 512 from sa2
        self.fp1 = PointNetFeaturePropagation(128, [128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn_conv1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 1, 1)

        if gating:
            self.context_gating = GatingContext(
                output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x, xyz):
        """
        x: [B, C, N] 特征张量
        xyz: [B, 1, N, 3] 原始点云

        Returns:
            vlad: [B, output_dim] VLAD 编码
            weights: [B, N, 1] 注意力权重
        """
        #print(f"NetVLAD input shapes - x: {x.shape}, xyz: {xyz.shape}")

        batch_size = x.size(0)

        # 1. 调整输入维度
        xyz = xyz.squeeze(1)  # [B, N, 3]
        x = x.transpose(1, 2).contiguous()  # [B, N, C]

        #print(f"After adjustment - xyz: {xyz.shape}, x: {x.shape}")

        try:
            # 2. PointNet++ 特征提取
            l1_xyz, l1_points = self.sa1(xyz, x)  # [B, N1, 3], [B, C1, N1]
            #print(f"After SA1 - l1_xyz: {l1_xyz.shape}, l1_points: {l1_points.shape}")

            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points.transpose(1, 2))  # [B, N2, 3], [B, C2, N2]
            #print(f"After SA2 - l2_xyz: {l2_xyz.shape}, l2_points: {l2_points.shape}")

            # 3. 特征传播，确保维度匹配
            l1_points_trans = l1_points.transpose(1, 2)  # [B, N1, C1]
            l2_points_trans = l2_points.transpose(1, 2)  # [B, N2, C2]

            l1_points = self.fp2(l1_xyz, l2_xyz, l1_points_trans, l2_points_trans)  # [B, C3, N1]
            #print(f"After FP2 - l1_points: {l1_points.shape}")

            l0_points = self.fp1(xyz, l1_xyz, None, l1_points.transpose(1, 2))  # [B, C4, N]
            #print(f"After FP1 - l0_points: {l0_points.shape}")

            # 4. 注意力权重计算
            score = self.drop1(F.relu(self.bn_conv1(self.conv1(l0_points))))  # [B, 128, N]
            score = self.conv2(score)  # [B, 1, N]
            score = score.transpose(1, 2)  # [B, N, 1]
            #print(f"Attention score shape: {score.shape}")

            # 5. NetVLAD 处理
            score = torch.sigmoid(score)  # [B, N, 1]
            weights = score  # 保存用于返回
            score = score.repeat(1, 1, self.cluster_size)  # [B, N, K]

            # 6. VLAD 特征聚合
            x_t = x  # [B, N, C]
            activation = torch.matmul(x_t, self.cluster_weights) # activation 表示每个点属于每个聚类中心的概率（或称为隶属度）。 # [B, N, K]

            if self.add_batch_norm:
                activation = activation.view(-1, self.cluster_size)
                activation = self.bn1(activation)
                activation = activation.view(batch_size, -1, self.cluster_size)
            else:
                activation = activation + self.cluster_biases

            activation = self.softmax(activation)  # [B, N, K]
            activation = torch.mul(activation, score)  # [B, N, K]

            # 7. VLAD 池化
            a_sum = activation.sum(-2, keepdim=True)  # [B, 1, K]
            a = a_sum * self.cluster_weights2  # a[b,f,k]表示：对于批次中的第b个样本，在特征维度f上，聚类中心k的全局期望激活强度。 # [B, C, K]

            activation = activation.transpose(2, 1)  # [B, K, N]
            vlad = torch.matmul(activation, x_t)# vlad[b, k, f] 表示批次中第 b 个样本中，所有点在特征维度 f 上属于第 k 个聚类中心的加权和。  # [B, K, C]
            vlad = vlad.transpose(2, 1)  # [B, C, K]
            vlad = vlad - a # 计算局部特征与全局期望的偏差，消除聚类中心权重本身的偏置影响

            # 8. 特征规范化 - 添加 contiguous()
            vlad = F.normalize(vlad, p=2, dim=1)  # 第一阶段归一化 （按特征通道）：确保每个聚类中心的特征向量具有单位范数，消除不同聚类中心之间的相对尺度差异。
            vlad = vlad.contiguous()  # 添加这一行
            vlad = vlad.view(batch_size, -1)  # [B, C*K]
            vlad = F.normalize(vlad, p=2, dim=1)  # 第二阶段归一化 （展开后整体）：控制整个描述符的全局范数，符合大多数检索系统对特征向量的标准化要求。

            # 9. 最终投影
            vlad = torch.matmul(vlad, self.hidden1_weights)  # [B, output_dim]
            vlad = self.bn2(vlad)

            # 10. 上下文门控（可选）
            if self.gating:
                vlad = self.context_gating(vlad)

            #print(f"Final output shapes - vlad: {vlad.shape}, weights: {weights.shape}")
            return vlad, weights

        except RuntimeError as e:
            print("\nError occurred in NetVLAD forward pass:")
            print(f"Current tensor shapes:")
            print(f"xyz: {xyz.shape}")
            print(f"x: {x.shape}")
            print(f"l1_xyz: {l1_xyz.shape if 'l1_xyz' in locals() else 'Not created'}")
            print(f"l1_points: {l1_points.shape if 'l1_points' in locals() else 'Not created'}")
            raise e


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation


class Flatten(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, input):
        return input.view(input.size(0), -1)


class STN3d(nn.Module):
    def __init__(self, num_points=2500, k=3, use_bn=True):
        super(STN3d, self).__init__()
        self.k = k
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1,1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1,1))
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)# 在(4096,1)的池化窗口内取最大值，也就是以一个点云文件为单位，找到这个点云文件的各个特征通道上的最大值。
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()

        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(
            1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points, k=3, use_bn=False)
        self.feature_trans = STN3d(num_points=num_points, k=64, use_bn=False)
        self.apply_feature_trans = feature_transform
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv3 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv4 = torch.nn.Conv2d(64, 128, (1, 1))
        self.conv5 = torch.nn.Conv2d(128, 1024, (1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024) # 对1024个通道进行标准化，对每个batch的数据进行标准化，使其均值为0，方差为1
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.num_points = num_points
        self.global_feat = global_feat
        self.max_pool = max_pool

    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = torch.matmul(torch.squeeze(x), trans)
        x = x.view(batchsize, 1, -1, 3)
        #x = x.transpose(2,1)
        #x = torch.bmm(x, trans)
        #x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        pointfeat = x
        if self.apply_feature_trans:
            f_trans = self.feature_trans(x)
            x = torch.squeeze(x)
            if batchsize == 1:
                x = torch.unsqueeze(x, 0)
            x = torch.matmul(x.transpose(1, 2), f_trans)
            x = x.transpose(1, 2).contiguous()
            x = x.view(batchsize, 64, -1, 1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        if not self.max_pool:
            return x
        else:
            x = self.mp1(x)
            x = x.view(-1, 1024)
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans


class PointNetVlad(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True, output_dim=1024):
        super(PointNetVlad, self).__init__()
        self.point_net = PointNetfeat(num_points=num_points, global_feat=global_feat,
                                    feature_transform=feature_transform, max_pool=max_pool)
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=num_points, cluster_size=64,
                                    output_dim=output_dim, gating=True, add_batch_norm=True,
                                    is_training=True)
        # 添加 max_pool 属性
        self.max_pool = max_pool

    def forward(self, x):
        #print(f"Input shape: {x.shape}")  # [B, 1, N, 3]

        f = self.point_net(x)  # 获取特征
        #print(f"PointNet output shape: {f.shape if not isinstance(f, tuple) else [t.shape for t in f]}")

        if isinstance(f, tuple):
            f = f[0]

        if not self.max_pool and len(f.shape) == 4:
            f = f.squeeze(-1)  # 移除最后的维度，得到 [B, C, N]
        #print(f"Feature shape before NetVLAD: {f.shape}")

        vlad, weights = self.net_vlad(f, x)
        return vlad, weights

from torchsummary import summary

if __name__ == '__main__':
    num_points = 4096

    # 创建示例输入数据
    sample_input = torch.randn(1, 1, num_points, 3).cuda()  # [batch_size, channels, num_points, xyz]

    # 初始化模型
    pnv = PointNetVlad(global_feat=True, feature_transform=True, max_pool=False,
                       output_dim=256, num_points=num_points).cuda()
    # 将模型设置为评估模式
    pnv.eval()  # 添加这一行

    # 打印模型结构
    print("Model structure:")
    print(pnv)

    # 打印参数数量
    total_params = sum(p.numel() for p in pnv.parameters())
    print(f'\nTotal parameters: {total_params:,}')

    # 测试前向传播
    print("\nTesting forward pass:")
    print(f'Input shape: {sample_input.shape}')
    with torch.no_grad():
        output, weights = pnv(sample_input)
        print(f'Output shape: {output.shape}')
        print(f'Weights shape: {weights.shape}')