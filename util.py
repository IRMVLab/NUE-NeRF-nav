from torch import nn
import torch.nn.functional as F
import torch as t
import numpy as np

mse2psnr = lambda x: -10. * t.log(x) / t.log(t.Tensor([10.]))
img2mse_uncert_alpha = lambda x, y, uncert, alpha, w : t.mean((1 / (2*(uncert+1e-9))) *((x - y) ** 2)) + 0.5*t.mean(t.log(uncert+1e-9)) + w * alpha.mean() + 4.0


def multi_l1_loss(output, target):
    def one_scale(output, target):
        b, _, h, w = output.size()
        target_scaled = F.interpolate(target, (h, w), mode='area')
        loss_ = 0
        loss_ += F.l1_loss(output, target_scaled)
        return loss_
    weights = [0.32, 0.08, 0.02, 0.01]
    loss = 0
    for out, weight in zip(output, weights):
        loss += weight * one_scale(out, target)
    return loss
    


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding = None):
    if padding == None:
        padding = (kernel_size-1)//2
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def deconv(in_planes, out_planes, kernel_size=4):  #0=s(i-1)-2p+k
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )

# def correlate(input1, input2):
#     out_corr = spatial_correlation_sample(input1,
#                                           input2,
#                                           kernel_size=1,
#                                           patch_size=21,
#                                           stride=1,
#                                           padding=0,
#                                           dilation_patch=2)
#     # collate dimensions 1 and 2 in order to be treated as a
#     # regular 4D tensor
#     b, ph, pw, h, w = out_corr.size()
#     out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1) #(batchsize, 441, H,W)
#     return F.leaky_relu_(out_corr, 0.1)

def adjust_learning_rate(initial_lr, lr_decay_step, episode, optimizer):
    if lr_decay_step > 0:
        learning_rate = 0.1 * initial_lr * (
                lr_decay_step - episode) / lr_decay_step + 0.1 * initial_lr
        if episode > lr_decay_step:
            learning_rate = 0.1 * initial_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    else:
        learning_rate = initial_lr
    return learning_rate

def ssim_loss(x,y):
    x = x.transpose(0, 2).transpose(1, 2).unsqueeze(0)
    y = y.transpose(0, 2).transpose(1, 2).unsqueeze(0)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, 3, 1)
    mu_y = F.avg_pool2d(y, 3, 1)
    sigma_x = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    return t.clamp((1 - ssim) / 2, 0, 1).mean()

def smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    pred_map = pred_map.unsqueeze(0).unsqueeze(0)
    loss = 0
    weight = 1.
    dx, dy = gradient(pred_map)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()) * weight
    return loss

def inter_sample(rays_dp,N_sample, half_dist,jitter=True, is_test = False):
    def jitter_fn(z_vals):
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = t.cat([mids, z_vals[..., -1:]], -1)
        lower = t.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        d1,d2 = z_vals.shape
        t_rand = t.rand(d1,d2).to(z_vals.device)
        return lower + (upper - lower) * t_rand

    if not hasattr(inter_sample, 'sample'):
        inter_sample.sample = N_sample
        inter_sample.half_dist = half_dist

    if not hasattr(inter_sample, 'z_vals') or not inter_sample.sample == N_sample\
            or not inter_sample.half_dist == half_dist:
        inter_sample.half_dist = half_dist
        inter_sample.sample = N_sample
        far = t.ones_like(rays_dp)*inter_sample.half_dist
        near = t.ones_like(rays_dp)*-1*inter_sample.half_dist
        t_N_vals = t.linspace(0., 1., steps=N_sample).to(rays_dp.device)
        inter_sample.z_vals = near * (1. - t_N_vals) + far * (t_N_vals)


    if not is_test:
        dp = rays_dp.clone()
        dp[dp < half_dist+0.05] = half_dist+0.05
        z_vals = inter_sample.z_vals + dp
        if jitter:
            z_vals = jitter_fn(z_vals)

        return z_vals
    else:
        near = np.ones_like(rays_dp) * near
        far = rays_dp + 0.3
        t_N_vals = np.linspace(0., 1., num=N_sample)
        z_vals = near * (1. - t_N_vals) + far * (t_N_vals)
        z_vals = jitter_fn(z_vals)
        return t.from_numpy(z_vals)

def generate_z_vals_and_depths(depth,N_sample = 64,half_dist=1.0,jitter=True):
    rays_dp = t.reshape(depth[0,-1, 0:-1:2,0:-1:2], [-1, 1])*10.
    z_vals = inter_sample(rays_dp, N_sample=N_sample, half_dist=half_dist, jitter=jitter)
    return depth[0,-1,0:-1:2,0:-1:2]*10., z_vals

def get_rays(H, W, K, c2w):
    if not hasattr(get_rays, 'i') and not hasattr(get_rays, 'j'):
        i, j = t.meshgrid(t.linspace(0, W-1, W), t.linspace(0, H-1, H))
        get_rays.i = i.t()
        get_rays.j = j.t()
    if not hasattr(get_rays, 'dirs'):
        get_rays.dirs = t.stack([-t.ones_like(get_rays.i), -(get_rays.i-K[0][2])/K[0][0], -(get_rays.j-K[1][2])/K[1][1]], -1)
    device = c2w.device
    dirs = get_rays.dirs.to(device)
    # Rotate ray directions from camera frame to the world frame
    rays_d = t.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return t.stack([rays_o, rays_d],dim=0) # [2,H,W,3]

def generate_rays_half(poses,H, W, K, z_vals):

    rays = t.stack([get_rays(H, W, K, p) for p in [poses]], 0)  # [1, ro+rd, H, W, 3]

    rays = t.transpose(rays, 1, 2)  # [1, H, ro+rd, W, 3]
    rays = t.transpose(rays, 2, 3)[0,0:-1:2,0:-1:2]  # [1, H, W, ro+rd, 3] = [1,H,W,2,3]

    rays = t.reshape(rays, (-1, 2, 3)).float()  # [H*W, ro+rd, 3] = [H*W,2,3]

    rays_o, rays_d = rays[:, 0], rays[:, 1]    # [H*W, 3]

    viewdirs = rays_d / t.norm(rays_d, dim=-1, keepdim=True)  # 将方向归一化

    pts = rays_o[..., None, :] + viewdirs[..., None, :] * z_vals[..., :, None]  # [N_rays, N_sample, 3]
    return pts.float(), viewdirs.float()

def minibatch(batch_size, model, pts, viewdirs):
    if len(pts) > batch_size:
        length = len(pts)
        raw, prd, uncertainty = [], [], []
        for i in range(0, length, batch_size):
            a, b, c = model(pts[i:i + batch_size], viewdirs[i:i + batch_size])
            raw.append(a)
            prd.append(b)
            uncertainty.append(c)
        return t.cat(raw, 0), t.cat(prd, 0), t.cat(uncertainty, 0)
    else:
        raw, prd, uncertainty = model(pts, viewdirs)
        return raw, prd, uncertainty

def render_pred(raw, prd, uncertainty, z_vals,H,W,is_flat=False):
    raw2alpha = lambda raw, dists: 1. - t.exp(-raw * dists)
    device = z_vals.device

    dists = z_vals[..., 1:] - z_vals[..., :-1]   # [N_rays, N_samples]
    dists = t.cat([dists, t.Tensor([1e10]).to(device).expand(dists[..., :1].shape)], -1)

    alpha_raw, rgb_emit, prd_emit, uncertainty_emit = raw[..., 0], raw[..., 1:], prd[..., :], uncertainty[..., :]

    alpha = raw2alpha(alpha_raw, dists)
    weights = alpha * t.cumprod(t.cat([t.ones((alpha.shape[0], 1)).to(device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    prd_map = t.sum(weights[..., None] * prd_emit, -2)
    uncertainty_map = t.sum(weights[..., None] * weights[..., None] * uncertainty_emit, -2)
    alpha = F.relu(alpha_raw).mean(-1).view(H, W)
    if is_flat:
        return prd_map, uncertainty_map, alpha
    else:
        return prd_map.view(H, W, -1), uncertainty_map.view(H, W, -1), alpha
# def render(raw, prd, z_vals,H,W,is_flat=False):
#     raw2alpha = lambda raw, dists: 1. - t.exp(-raw * dists)
#     device = z_vals.device
#
#     dists = z_vals[..., 1:] - z_vals[..., :-1]   # [N_rays, N_samples]
#     dists = t.cat([dists, t.Tensor([1e10]).to(device).expand(dists[..., :1].shape)], -1)
#
#     alpha_raw, rgb_emit, prd_emit = raw[..., 0], raw[..., 1:], prd[..., :]
#
#     alpha = raw2alpha(alpha_raw, dists)
#     T = t.cumprod(t.cat([t.ones((alpha.shape[0], 1)).to(device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
#     alpha = alpha * T
#     rgb_map = t.sum(alpha[..., None] * rgb_emit, -2)
#     depth_map = t.sum(alpha * z_vals, -1)
#     prd_map = t.sum(alpha[..., None] * prd_emit, -2).transpose(0, 1)
#     if is_flat:
#         return rgb_map, depth_map, prd_map
#     else:
#         return rgb_map.view(H, W, -1), depth_map.view(H, W), prd_map.view(1, -1, H, W)
def render(raw, uncertainty, z_vals, H, W, is_flat=False):
    raw2alpha = lambda raw, dists: 1. - t.exp(-raw * dists)
    device = z_vals.device

    dists = z_vals[..., 1:] - z_vals[..., :-1]   # [N_rays, N_samples]
    dists = t.cat([dists, t.Tensor([1e10]).to(device).expand(dists[..., :1].shape)], -1)

    alpha_raw, rgb_emit, uncertainty_emit = raw[..., 0], raw[..., 1:], uncertainty[..., :]

    alpha = raw2alpha(alpha_raw, dists)
    weights = alpha * t.cumprod(t.cat([t.ones((alpha.shape[0], 1)).to(device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    uncertainty_map = t.sum(weights[..., None] * weights[..., None] * uncertainty_emit, -2)

    # prd_map = t.sum(alpha[..., None] * prd_emit, -2)
    rgb_map = t.sum(weights[..., None] * rgb_emit, -2)
    depth_map = t.sum(weights * z_vals, -1)
    alpha = F.relu(alpha_raw).mean(-1).view(H, W)
    if is_flat:
        return rgb_map, depth_map, uncertainty_map, alpha
    else:
        return rgb_map.view(H, W, -1), depth_map.view(H, W), uncertainty_map.view(H, W, -1), alpha


def loss(rgb_map, depth_map, rgb, depth, is_ssim, is_smooth):
    if is_ssim:
        h,w,_ = rgb_map.shape
    loss_ssim = ssim_loss(rgb_map.view(h,w,-1),rgb.view(h,w,-1)) if is_ssim else 0
    loss_smooth = smooth_loss(depth_map) if is_smooth else 0
    loss_rgb_mse = F.mse_loss(rgb_map.float(),rgb)
    # loss_pred = F.l1_loss(pred, prd_truth)
    loss = 1.*loss_rgb_mse + 1.*F.mse_loss(depth_map.float(),depth) + \
           0.05*loss_ssim+0.15*loss_smooth
    # loss = 0.85 * F.mse_loss(depth_map.float(),depth) + 0.15 * loss_ssim + 0.15 * loss_smooth
    return loss, mse2psnr(loss_rgb_mse.cpu())

def Quat2Rotation(x,y,z,w):
    l1 = np.stack([1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y + 2 * w * z, 2 * x * z - 2 * w * y],axis=0)
    l2 = np.stack([2 * x * y - 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z + 2 * w * x],axis=0)
    l3 = np.stack([2 * x * z + 2 * w * y, 2 * y * z - 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2],axis=0)
    T_w = np.stack([l1,l2,l3],axis=0)
    return T_w

def gaussian2d(x, y, mux, muy, sigmax, sigmay, rho=0):
    """
    计算二维高斯函数在(x, y)处的值
    :param x: 自变量x
    :param y: 自变量y
    :param mux: x方向均值
    :param muy: y方向均值
    :param sigmax: x方向标准差
    :param sigmay: y方向标准差
    :param rho: x和y的相关系数
    :return: 二维高斯函数在(x, y)处的值
    """
    z = ((x - mux) / sigmax) ** 2 + ((y - muy) / sigmay) ** 2 - 2 * rho * (x - mux) * (y - muy) / (sigmax * sigmay)
    return 1 / (2 * np.pi * sigmax * sigmay * np.sqrt(1 - rho ** 2)) * np.exp(-z / (2 * (1 - rho ** 2)))

def cosine_similarity(x, y):
    """
    计算两个张量之间的余弦相似度。

    Args:
        x: 第一个张量，形状为 (n, d)。
        y: 第二个张量，形状为 (m, d)。

    Returns:
        一个形状为 (n, m) 的张量，表示 x 中每个向量与 y 中每个向量之间的余弦相似度。
    """
    # 对 x 和 y 中的向量进行 L2 归一化
    y = y.unsqueeze(0)
    x = x/t.norm(x)
    y = y/t.norm(y)
    # 计算余弦相似度
    similarity = 1-t.matmul(x, y.T)

    return similarity

def render_heatmap(H, W, K, c2w, robot_pos, target_pos):
    device = c2w.device
    print(device)
    w2c = t.inverse(c2w).to(device)
    distance = np.linalg.norm(target_pos - robot_pos)
    ray_dir = t.from_numpy(target_pos - robot_pos).float() / distance

    # 将光线方向表示为齐次坐标形式
    ray_dir_hom = t.cat([ray_dir.unsqueeze(0), t.ones((1, 1))], dim=1).to(device)
    # 将光线方向从世界坐标系转换到相机坐标系
    ray_dir_c_hom = t.matmul(ray_dir_hom, w2c)
    # 最终的光线方向为ray_dir_c，形状为(1,3)
    ray_dir_c = ray_dir_c_hom[:, :3] / (ray_dir_c_hom[:, 3:] + 0.01 ** 2)
    ray_dir_c[0, 0] = -ray_dir_c[0, 0]

    if not hasattr(render_heatmap, 'i') and not hasattr(render_heatmap, 'j'):
        i, j = t.meshgrid(t.linspace(0, W - 1, W), t.linspace(0, H - 1, H))
        render_heatmap.i = i.t()
        render_heatmap.j = j.t()
    if not hasattr(render_heatmap, 'dirs'):
        render_heatmap.dirs = t.stack([-t.ones_like(render_heatmap.i), -(render_heatmap.i - K[0][2]) / K[0][0], -(render_heatmap.j - K[1][2]) / K[1][1]],-1)  # get_rays.dirs:[180, 240, 3]

    ray_dir_c = ray_dir_c.to(device)
    dirs = render_heatmap.dirs.to(device)
    distances = t.zeros(H, W)
    print(distances)
    for i in range(H):
        for j in range(W):
            distances[i, j] = cosine_similarity(ray_dir_c, dirs[i, j])
    distances = distances.reshape(H, W)
    min_index = t.argmin(distances)
    target_pixel_i = min_index // W
    target_pixel_j = min_index - W * target_pixel_i
    heatmap = t.zeros(H, W)
    print(heatmap)
    for i in range(H):
        for j in range(W):
            heatmap[i, j] = 4000*gaussian2d(i, j, target_pixel_i, target_pixel_j, 4*distance, 4*distance)

    heatmap = heatmap[0:-1:2, 0:-1:2]  # heatmap:[90, 120]

    return heatmap
