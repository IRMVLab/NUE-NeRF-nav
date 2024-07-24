import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from util import *
from torch.optim import Adam
import random
from random import choice
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from trainer import writeSummary
from copy import deepcopy
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    pass
from time import time
# torch.manual_seed(144152)
# torch.cuda.manual_seed_all(144152)
# np.random.seed(144152)
# random.seed(144152)

class EmbedFunction(torch.nn.Module):
    def __init__(self, p_fn, freq):
        super(EmbedFunction, self).__init__()
        self.p_fn = p_fn
        self.freq = freq

    def forward(self, x):
        return self.p_fn(x * self.freq)

def Quat2Rotation(x,y,z,w):
    l1 = torch.stack([1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y + 2 * w * z, 2 * x * z - 2 * w * y], dim=0)
    l2 = torch.stack([2 * x * y - 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z + 2 * w * x], dim=0)
    l3 = torch.stack([2 * x * z + 2 * w * y, 2 * y * z - 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2], dim=0)
    T_w = torch.stack([l1, l2, l3], dim=0)
    return T_w

def Rotation2Quat(pose):
    m11,m22,m33 = pose[0][0],pose[1][1],pose[2][2]
    m12,m13,m21,m23,m31,m32 = pose[0][1],pose[0][2],pose[1][0],pose[1][2],pose[2][0],pose[2][1]
    x,y,z,w = torch.sqrt(m11-m22-m33+1)/2,torch.sqrt(-m11+m22-m33+1)/2,torch.sqrt(-m11-m22+m33+1)/2,torch.sqrt(m11+m22+m33+1)/2
    Quat_ = torch.tensor([
        [x,(m12+m21)/(4*x),(m13+m31)/(4*x),(m23-m32)/(4*x)],
        [(m12+m21)/(4*y),y,(m23+m32)/(4*y),(m31-m13)/(4*y)],
        [(m13 + m31) / (4 * z), (m23 + m32) / (4 * z), z,(m12 - m21) / (4 * z)],
        [(m23 - m32) / (4 * w), (m31 - m13) / (4 * w), (m12 - m21) / (4 * w),w]
    ], dtype=torch.float32)
    _,index = torch.tensor([x,y,z,w]).max(dim=0)
    Quat = Quat_[index.item()]
    return Quat

def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions
def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))
def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def iden(x):
    return x

class NeRF_pi(nn.Module):
    def __init__(self, input_dim, W=64, pos_multires=10, dir_multires=4):
        super(NeRF_pi,self).__init__()
        self.input_dim = input_dim * (pos_multires * 2 + 1)
        self.input_dir_dim = input_dim * (dir_multires * 2 + 1)
        self.pos_freq_bands = (2. ** torch.linspace(0., pos_multires - 1, steps=pos_multires)) * torch.pi
        self.dir_freq_bands = (2. ** torch.linspace(0., dir_multires - 1, steps=dir_multires)) * torch.pi
        # self.pos_embed_fn = self.embed(pos_freq_bands)
        # self.dir_embed_fn = self.embed(dir_freq_bands)
        # self.input_dim = 50
        self.W = W
        self.part1 = nn.Sequential(
            nn.Linear(self.input_dim, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU()
        )
        self.part2 = nn.Sequential(
            nn.Linear(self.input_dim+W, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
        )
        self.part3 = nn.Sequential(
            nn.Linear(W + self.input_dir_dim, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
        )
        self.alpha_linear = nn.Sequential(
            nn.Linear(W, 1),
            nn.ReLU(),
        )
        self.rgb_linear = nn.Sequential(
            nn.Linear(W, 3),
            nn.ReLU()
        )
        self.uncertainty_linear = nn.Sequential(
            nn.Linear(W, 1),
            nn.ReLU()
        )
        self.act_uncertainty = nn.Softplus()

    # def embed(self,freq_bands):
    #     embed_fns=[iden]
    #     for freq in freq_bands:
    #         for p_fn in [torch.sin, torch.cos]:
    #             def _func(x, p_fn=p_fn, freq=freq):
    #                 return p_fn(x * freq)
    #
    #             embed_fns.append(_func)
    #     return embed_fns

    def get_embed(self,x, freq_bands):
        with torch.no_grad():
            x_ = torch.cat([x*freq for freq in freq_bands], -1)
            x_ = torch.cat([fn(x_) for fn in [torch.sin, torch.cos]], -1).float()
            return torch.cat([x, x_], -1).float()


    def forward(self, pts, viewdirs):
        N_ray, N_sample, _ = pts.shape
        viewdirs = viewdirs[:, None].expand(pts.shape).reshape(-1, 3)
        pts = pts.view(-1, 3)
        gamma = self.get_embed(pts, self.pos_freq_bands)
        dirs = self.get_embed(viewdirs, self.dir_freq_bands)
        out = self.part1(gamma)
        out = torch.cat([out, gamma], -1)
        out1 = self.part2(out)
        alpha = self.alpha_linear(out1).view(N_ray, N_sample, -1)
        uncertainty = self.act_uncertainty(self.uncertainty_linear(out1).view(N_ray, N_sample, -1))
        out2 = self.part3(torch.cat([out1, dirs], -1)).view(N_ray, N_sample, -1)
        rgb = self.rgb_linear(out2).view(N_ray, N_sample, -1)
        return torch.cat([alpha, rgb], dim=2), out2, uncertainty


class resnet_block(nn.Module):
    def __init__(self, in_channel, out_channel, alpha=1):
        super(resnet_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=alpha, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(0.1, True)
        )
        self.byp = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=alpha, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(0.1, True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        return self.out(conv1 + self.byp(x))


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps (B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        return out

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Attention(nn.Module):
    def __init__(self, input_size):
        super(Attention, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(0.1, True),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(0.1, True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        x = self.linear(h)
        x = self.sigmoid(x)
        h = h * x
        return h


# 感知特征提取网络
class bypath(nn.Module):
    def __init__(self):
        super(bypath, self).__init__()
        self.front_conv_1 = nn.Sequential(
            resnet_block(4, 64, 2),    # [1, 64, 45, 60]
            resnet_block(64, 128, 2),  # [1, 128, 12, 15]
        )
        self.attention = Self_Attn(128)
        self.front_conv_2 = nn.Sequential(
            resnet_block(128, 256, 2),  # [1, 256, 3, 4]
            nn.Flatten()
        )
        self.out = nn.Sequential(
            nn.Linear(256*12, 1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, observation):
        h = self.front_conv_1(observation)
        h = self.attention(h)
        h = self.front_conv_2(h)
        out = self.out(h)  # out:[1, 256]
        return out

class Feature_Extra(nn.Module):
    def __init__(self):
        super(Feature_Extra, self).__init__()
        self.front_conv = nn.Sequential(
            resnet_block(in_channel=67, out_channel=64, alpha=1),  # [1, 64, 45, 60]
            resnet_block(in_channel=64, out_channel=32, alpha=1)  # [1, 32, 23, 30]
        )
        self.attention = Self_Attn(32)

    def forward(self, prd):
        prd = prd.transpose(2, 3).transpose(1, 2)
        prd = self.front_conv(prd)
        prd = self.attention(prd)
        return prd

class Prd_Linear(nn.Module):
    def __init__(self):
        super(Prd_Linear, self).__init__()
        self.front_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, True)
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(768, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.angle_pred_linear = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 1)
        )
        #self.norm = nn.BatchNorm1d(64)


    def forward(self, prd):
        prd = self.front_conv(prd)  # prd:[1, 16, 6, 8]
        prd = self.flatten(prd)
        h = self.linear(prd)  # h:[1, 64]
        out = h
        #out = self.norm(h)
        pred_angle = self.angle_pred_linear(h)
        return out, pred_angle


class Exploration_Net(nn.Module):
    def __init__(self):
        super(Exploration_Net, self).__init__()
        self.front_conv = nn.Sequential(
            resnet_block(in_channel=1, out_channel=8, alpha=2),  # [1, 16, 23, 30]
            resnet_block(in_channel=8, out_channel=16, alpha=2)  # [1, 16, 6, 7]
        )

        self.attention = Self_Attn(16)
        self.flatten = nn.Flatten()
        self.out = nn.Sequential(
            nn.Linear(in_features=768, out_features=512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=256, out_features=64),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, uncertainty_map):
        h = self.front_conv(uncertainty_map)
        h = self.attention(h)
        h = self.flatten(h)
        out = self.out(h)
        return out

class E2E_model_qkv(nn.Module):
    def __init__(self, action_space):
        super(E2E_model_qkv, self).__init__()
        self.bypath = bypath()
        self.exploration_net = Exploration_Net()
        self.extra_net = Feature_Extra()
        self.pred_net = Prd_Linear()
        self.attention_net = Attention(256+64+64)

        self.policy_net = nn.Sequential(
            nn.Linear(256+64+64, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, action_space)
        )

    def forward(self, observation, out_pred=None, uncertainty_map=None, type='gathering'):
        if type == 'gathering':
            uncertainty_map = uncertainty_map.transpose(2, 3).transpose(1, 2)

            t0 = time()
            uncert_pred = self.exploration_net(uncertainty_map)  # uncert_pred:[1, 256]
            h = self.extra_net(out_pred)
            fc, pred_angle = self.pred_net(h)

            t1 = time()

            out_bypath = self.bypath(observation)  # [1, 256]
            h = self.attention_net(torch.cat([uncert_pred, fc, out_bypath], dim=1))
            pi = self.policy_net(h)

            t2 = time()

            dt_cog = t1 - t0
            dt_pol = t2 - t1
            return F.softmax(pi, dim=1), dt_cog, dt_pol
        else:
            uncertainty_map = uncertainty_map.transpose(2, 3).transpose(1, 2)
            uncert_pred = self.exploration_net(uncertainty_map)  # uncert_pred:[1, 256]
            h = self.extra_net(out_pred)
            fc, pred_angle = self.pred_net(h)
            out_bypath = self.bypath(observation)  # [1, 256]
            h = self.attention_net(torch.cat([uncert_pred, fc, out_bypath], dim=1))
            pi = self.policy_net(h)
            return F.softmax(pi, dim=1), pred_angle
