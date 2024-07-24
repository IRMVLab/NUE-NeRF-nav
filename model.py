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

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

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

        return out, attention

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
        self.channel_attention = ChannelAttention(128)
        self.spatial_attention = SpatialAttention()
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
        #self.norm = nn.BatchNorm1d(256)

    def forward(self, observation):
        h = self.front_conv_1(observation)
        #x = self.channel_attention(h)
        #y = self.spatial_attention(h)
        #h = h * x * y
        h = self.front_conv_2(h)
        out = self.out(h)  # out:[1, 256]
        #out = self.norm(out)
        return out

class Feature_Extra(nn.Module):
    def __init__(self):
        super(Feature_Extra, self).__init__()
        self.front_conv = nn.Sequential(
            resnet_block(in_channel=67, out_channel=64, alpha=1),  # [1, 64, 45, 60]
            resnet_block(in_channel=64, out_channel=32, alpha=1)  # [1, 32, 23, 30]
        )
        self.channel_attention = ChannelAttention(32)
        self.spatial_attention = SpatialAttention()

    def forward(self, prd):
        prd = prd.transpose(2, 3).transpose(1, 2)
        prd = self.front_conv(prd)
        #x = self.channel_attention(prd)
        #y = self.spatial_attention(prd)
        #prd = prd * x * y  # prd:[1, 32, 23, 30]
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

class Deconv_Net(nn.Module):
    def __init__(self):
        super(Deconv_Net, self).__init__()
        self.front_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, True)
        )
        self.deconv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2)  # 转置卷积层1
        self.deconv2 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2)  # 转置卷积层2
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2)  # 转置卷积层3

    def forward(self, prd):
        prd = self.front_conv(prd)
        prd = self.deconv1(prd)  # 转置卷积层1
        prd = self.deconv2(prd)  # 转置卷积层2
        prd = prd[:, :, 2:-2, 1:-1]
        heat_map = self.deconv3(prd)  # 转置卷积层3
        heat_map = heat_map[:, :, 2:-3, 1:-2]
        return heat_map


class Exploration_Net(nn.Module):
    def __init__(self):
        super(Exploration_Net, self).__init__()
        self.front_conv = nn.Sequential(
            resnet_block(in_channel=1, out_channel=8, alpha=2),  # [1, 16, 23, 30]
            resnet_block(in_channel=8, out_channel=16, alpha=2)  # [1, 16, 6, 7]
        )
        self.channel_attention = ChannelAttention(16)
        self.spatial_attention = SpatialAttention()
        self.flatten = nn.Flatten()
        self.out = nn.Sequential(
            nn.Linear(in_features=768, out_features=512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=256, out_features=64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        #self.norm = nn.BatchNorm1d(64)

    def forward(self, uncertainty_map):
        h = self.front_conv(uncertainty_map)
        #x = self.channel_attention(h)
        #y = self.spatial_attention(h)
        #h = h * x * y
        h = self.flatten(h)
        out = self.out(h)
        #out = self.norm(out)
        return out

class E2E_model(nn.Module):
    def __init__(self, action_space):
        super(E2E_model, self).__init__()
        self.bypath = bypath()
        self.exploration_net = Exploration_Net()
        self.extra_net = Feature_Extra()
        self.pred_net = Prd_Linear()
        # self.deconv = Deconv_Net()
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

class E2E_model_without_exploration(nn.Module):
    def __init__(self, action_space):
        super(E2E_model_without_exploration, self).__init__()
        self.bypath = bypath()
        self.extra_net = Feature_Extra()
        self.pred_net = Prd_Linear()
        self.attention_net = Attention(256+64)

        self.policy_net = nn.Sequential(
            nn.Linear(256+64, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, action_space)
        )

    def forward(self, observation, out_pred=None, uncertainty_map=None, type='gathering'):
        if type == 'gathering':
            h = self.extra_net(out_pred)
            fc, pred_angle = self.pred_net(h)
            out_bypath = self.bypath(observation)  # [1, 256]
            h = self.attention_net(torch.cat([fc, out_bypath], dim=1))
            pi = self.policy_net(h)
            return F.softmax(pi, dim=1)
        else:
            h = self.extra_net(out_pred)
            fc, pred_angle = self.pred_net(h)
            out_bypath = self.bypath(observation)  # [1, 256]
            h = self.attention_net(torch.cat([fc, out_bypath], dim=1))
            pi = self.policy_net(h)
            return F.softmax(pi, dim=1), pred_angle


class E2E_model_only_exploration(nn.Module):
    def __init__(self, action_space):
        super(E2E_model_only_exploration, self).__init__()
        self.bypath = bypath()
        self.exploration_net = Exploration_Net()
        self.attention_net = Attention(128 * 2)

        self.policy_net = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, action_space)
        )

        self.value_net = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 1)
        )

    def forward(self, observation, uncertainty_map=None, type='gathering'):
        if type == 'gathering':
            uncertainty_map = uncertainty_map.transpose(2, 3).transpose(1, 2)
            uncert_pred = self.exploration_net(uncertainty_map)  # uncert_pred:[1, 256]
            out_bypath = self.bypath(observation)  # [1, 256]
            h = self.attention_net(torch.cat([out_bypath, uncert_pred], dim=1))
            pi = self.policy_net(h)
            return F.softmax(pi, dim=1)
        else:
            uncertainty_map = uncertainty_map.transpose(2, 3).transpose(1, 2)
            uncert_pred = self.exploration_net(uncertainty_map)  # uncert_pred:[1, 256]
            out_bypath = self.bypath(observation)  # [1, 256]
            h = self.attention_net(torch.cat([out_bypath, uncert_pred], dim=1))
            pi = self.policy_net(h)
            value = self.value_net(h)
            return F.softmax(pi, dim=1), value


class NeRF_proc():
    def __init__(self, nerf_tmp, device, nerf_list, N_sample=64):
        super(NeRF_proc,  self).__init__()
        self.nerf = nerf_tmp
        self.feature_t = None
        self.N_sample = N_sample
        self.half_dist = 1.0
        self.jitter = True
        self.device = device
        self.nerf = self.nerf.to(self.device)
        self.nerf_list = nerf_list
        with open('cameras.txt', 'r') as f:
            K = f.readline()
            K = K.split(' ')
            self.H, self.W = int(K[-2]), int(K[-1])
            self.K = np.array([
                [float(K[0]), 0, float(K[2])],
                [0, float(K[1]), float(K[3])],
                [0, 0, 1]
            ], dtype=np.float32)

    def change_target(self, rgb_t):
        with torch.no_grad():
            self.feature_t = rgb_t[0:-1:2, 0:-1:2, :]

    def memory_process(self, drgb, pose, lock, queue, step, nerf_batch=10800, other_device=None):
        # t0=time()
        depth, rgb = drgb[:, 0:1], drgb[0, 1:, 0:-1:2, 0:-1:2].transpose(0, 1).transpose(1, 2)
        depths, z_vals = generate_z_vals_and_depths(depth, self.N_sample, self.half_dist, self.jitter)
        pts, viewdirs = generate_rays_half(pose, self.H, self.W, self.K, z_vals)
        # t1=time()
        # dt0=t1-t0
        if not len(self.nerf_list) == 0:
            lock.acquire()  # 尝试获取锁，获取成功则继续执行代码，否则阻塞
            _state = self.nerf_list[-1]  # 获取nerf_list中最后一个元素作为当前_state
            self.nerf.load_state_dict(_state)  # 通过路径_state加载模型
            self.nerf = self.nerf.to(self.device)
            self.nerf_list[:] = []  # 清空nerf_list
            lock.release()  # 释放锁，以便其他线程可以获取到这个锁并继续执行相应的代码

        raw, out2, uncertainty = minibatch(nerf_batch, self.nerf, pts, viewdirs)
        # t0 = time()
        pred, uncertainty_map, alpha = render_pred(raw, out2, uncertainty, z_vals, self.H // 2, self.W // 2, is_flat=False)  #[H, W, 64]
        # t1 = time()
        # dt1=t1-t0

        # 每隔四步将[pts, viewdirs, rgb, depths, z_vals]放入queue队列
        if step % 4 == 0:
            if other_device==None:
                queue.put([pts, viewdirs, rgb, depths, z_vals])
            else:
                queue.put([pts.to(other_device), viewdirs.to(other_device), rgb.to(other_device), depths.to(other_device), z_vals.to(other_device)])

        return torch.cat([pred, self.feature_t], dim=-1).unsqueeze(0), uncertainty_map, alpha


def nerf_reset(nerf, lock, nerf_list):
    for layer in nerf.modules():
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)  # 对线性层权重进行初始化
            layer.bias.data.fill_(0)  # 偏置设为0
    lock.acquire()
    nerf = nerf.cpu()
    if not len(nerf_list) == 0:
        nerf_list[:] = []
    nerf_list.append(deepcopy(nerf.state_dict()))
    lock.release()


def nerf_train(nerf, device, lock, queue, nerf_list, reset_list, child_conn):
    nerf = nerf.to(device)
    nerf.train()
    optimizer = Adam(nerf.parameters(), lr=0.001)
    data_cache = []
    nerf_batch = 10800
    count = 0
    last_undate = 0
    while True:
        if not queue.empty():
            data = queue.get()
            raw, out2, uncertainty = minibatch(nerf_batch, nerf, data[0], data[1])
            rgb_map, depth_map, uncertainty_map, alpha = render(raw, uncertainty, data[4], data[2].shape[0], data[2].shape[1], is_flat=False)
            loss_t = img2mse_uncert_alpha(rgb_map, data[2], uncertainty_map, alpha, 0.01)
            # loss_t, psnr = loss(rgb_map, depth_map, data[2], data[3], True, True)
            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()
            count += 1
            data_cache.append(data)
            if len(data_cache) > 100:
                data_cache = data_cache[20:]  # 清空data_cache中前20个元素
                torch.cuda.empty_cache()  # 清空GPU缓存

            if count-last_undate > 5:
                lock.acquire()
                nerf = nerf.cpu()
                if not len(nerf_list) == 0:
                    nerf_list[:] = []  # 清空nerf_list
                nerf_list.append(deepcopy(nerf.state_dict()))  # 将nerf模型参数保存到nerf_list中
                lock.release()
                nerf = nerf.to(device)
                last_undate = count

        if len(data_cache) > 0:
            pts, viewdirs, images, depths, z_vals = choice(data_cache)
            raw, out2, uncertainty = minibatch(nerf_batch, nerf, pts, viewdirs)
            rgb_map, depth_map, uncertainty_map, alpha = render(raw, uncertainty, z_vals, images.shape[0], images.shape[1], is_flat=False)
            loss_t = img2mse_uncert_alpha(rgb_map, data[2], uncertainty_map, alpha, 0.01)
            # loss_t, psnr = loss(rgb_map, depth_map, data[2], data[3], True, True)
            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()
            count += 1

        if reset_list[-1]:
            count = 0
            last_undate = 0
            del data_cache
            data_cache = []
            while not queue.empty():  ### 清空queue
                queue.get()
            reset_list[-1] = False
            # torch.cuda.empty_cache()
            nerf_reset(nerf, lock, nerf_list)  # nerf参数重置
            nerf = nerf.to(device)
            child_conn.send('nerf reset ok')


def nerf_train_for_test(nerf, device, lock, queue, nerf_list, reset_list, child_conn):
    nerf = nerf.to(device)
    nerf.train()
    optimizer = Adam(nerf.parameters(), lr=0.001)
    data_cache = []
    nerf_batch = 10800
    count = 0
    last_undate = 0
    nerf_train_dt = 0
    nerf_train_count = 0
    while True:
        if not queue.empty():
            data = queue.get()
            raw, out2, uncertainty = minibatch(nerf_batch, nerf, data[0], data[1])
            rgb_map, depth_map, uncertainty_map, alpha = render(raw, uncertainty, data[4], data[2].shape[0], data[2].shape[1], is_flat=False)
            loss_t = img2mse_uncert_alpha(rgb_map, data[2], uncertainty_map, alpha, 0.01)
            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()
            count += 1
            data_cache.append(data)
            if len(data_cache) > 100:
                data_cache = data_cache[20:]
                torch.cuda.empty_cache()

            if count-last_undate > 5:
                lock.acquire()
                nerf = nerf.cpu()
                if not len(nerf_list) == 0:
                    nerf_list[:] = []
                nerf_list.append(deepcopy(nerf.state_dict()))
                lock.release()
                nerf = nerf.to(device)
                last_undate = count

        if len(data_cache) > 0:
            t0 = time()
            pts, viewdirs, images, depths, z_vals = choice(data_cache)
            raw, out2, uncertainty = minibatch(nerf_batch, nerf, pts, viewdirs)
            rgb_map, depth_map, uncertainty_map, alpha = render(raw, uncertainty, z_vals, images.shape[0],
                                                                images.shape[1], is_flat=False)
            loss_t = img2mse_uncert_alpha(rgb_map, data[2], uncertainty_map, alpha, 0.01)
            # loss_t, psnr = loss(rgb_map, depth_map, data[2], data[3], True, True)
            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()
            t1 = time()
            nerf_train_dt += t1-t0
            nerf_train_count += 1
            count += 1

        # 一个episode结束，重置nerf
        if reset_list[-1]:
            count = 0
            last_undate = 0
            del data_cache
            data_cache = []
            print('nerf train', nerf_train_dt/nerf_train_count)
            nerf_train_dt = 0
            nerf_train_count = 0
            while not queue.empty():  ### 清空queue
                queue.get()
            reset_list[-1] = False
            # torch.cuda.empty_cache()
            nerf_reset(nerf, lock, nerf_list)
            nerf = nerf.to(device)
            child_conn.send('nerf reset ok')


def model_train(model, device, lock, source, summary_path, model_path, _flag):
    model = model.to(device)
    model.train()
    init_lr = 1e-3
    optimizer = Adam(model.parameters(), lr=init_lr)
    writer = SummaryWriter(summary_path)
    flag = False
    num = 0
    #batchSize = 128
    batchSize = 256
    total_p, total_t = 0, 0
    _count = 0
    episode = 0
    stats = {'policy_loss': [], 'pred_loss': [], 'learning_rate': []}
    print(len(source) > 0)
    while True:
        # 当source中有数据时
        if len(source) > 0:
            lock.acquire()  # 获取锁，确保只有一个进程可以访问source列表
            _action = np.array([tt.action for tt in source], dtype=np.long)
            indexl = np.where(_action == 1)[0]
            indexr = np.where(_action == 2)[0]
            indexf = np.where(_action == 0)[0]
            mean = ((len(indexl) + len(indexr))*5) // 6
            if len(indexf) > mean:
                indexf = np.random.choice(indexf, mean)  # 从直行动作的索引中随机选择一部分数据，使得直行动作的数据量接近平均值
            index_ = np.concatenate([indexl, indexf, indexr], 0)  # 将左转、直行和右转的索引连接起来
            action = []
            state = []
            out_pred = []
            label = []
            uncertainty = []
            for _i in range(len(source)):
                if _i in index_:
                    action.append(source[_i].action)
                    state.append(source[_i].state)
                    out_pred.append(source[_i].prd_map)
                    label.append(source[_i].label)
                    uncertainty.append(source[_i].uncertainty_map)

            action = torch.from_numpy(np.array(action, dtype=np.long)).view(-1, 1).to(device)
            state = torch.from_numpy(np.concatenate(state, 0)).float().to(device)
            out_pred = torch.cat(out_pred, dim=0).to(device)
            label = torch.from_numpy(np.stack(label, 0)).to(device)
            uncertainty = torch.from_numpy(np.concatenate(uncertainty, 0)).float().to(device)
            flag = True
            num = label.shape[0]  # 将num设置为标签数据的数量
            print('length:', num, len(indexl), len(indexr), len(indexf))
            source[:] = []  # 获取数据后清空source
            lock.release()
            torch.cuda.empty_cache()  # 清空GPU缓存
        if flag:
            _count = 0
            total_p, total_t = 0, 0
            # 随机取batch
            for index in BatchSampler(SubsetRandomSampler(range(num)), batchSize, False):
                action_prob, pred = model(observation=state[index], out_pred=out_pred[index], uncertainty_map=uncertainty[index], type='training')
                action_loss = F.cross_entropy(action_prob, action[index].view(-1))  # 动作之间交叉熵
                #theta_loss = F.l1_loss(pred, label[index])
                #loss = action_loss + theta_loss
                loss = action_loss
                optimizer.zero_grad()  # Delete old gradients
                loss.backward()  # Perform backward step to compute new gradients
                nn.utils.clip_grad_norm_(model.parameters(), 0.6)  # Clip gradients
                optimizer.step()  # Perform training step based on gradients
                total_p += action_loss
                #total_t += theta_loss #累加每个批次的动作损失和预测损失
                _count += 1
            episode += 1
            lr = adjust_learning_rate(init_lr, 1e5, episode, optimizer)
            stats['policy_loss'].append(total_p.cpu().item()/_count)
            #stats['pred_loss'].append(total_t.cpu().item()/_count)
            stats['learning_rate'].append(lr)
            writeSummary(writer, stats, episode)
            if episode % 1000 == 0:
                torch.save(model.state_dict(), model_path + str(episode) + '.pkl')