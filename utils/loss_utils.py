#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask

def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    # grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    # grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=0.0).squeeze()
    return grad_img



def get_normal_diff(normal):
    _, hd, wd = normal.shape 
    bottom_point = normal[..., 2:hd,   1:wd-1]
    top_point    = normal[..., 0:hd-2, 1:wd-1]
    right_point  = normal[..., 1:hd-1, 2:wd]
    left_point   = normal[..., 1:hd-1, 0:wd-2]
    grad_img_x = 1 - (right_point * left_point).sum(dim=0,keepdim=True)
    grad_img_y = 1 - (top_point * bottom_point).sum(dim=0,keepdim=True)
    # grad_img_x = torch.where(grad_img_x > 0, 1 - grad_img_x, grad_img_x)
    # grad_img_y = torch.where(grad_img_y > 0, 1 - grad_img_y, grad_img_y)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img = torch.mean(grad_img, dim=0)
    # grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    # grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=0.0).squeeze()
    # center_point = normal[..., 1:hd-1, 1:wd-1]
    # diff_img_x = 1 - (center_point * left_point).sum(dim=0,keepdim=True)
    # diff_img_y = 1 - (center_point * bottom_point).sum(dim=0,keepdim=True)
    # diff_img = (diff_img_x + diff_img_y) / 2
    
    return grad_img 

def local_plane_loss(depth, normal, camera):
    _, hd, wd = depth.shape 
    # breakpoint()
    points = depth * camera.get_rays().permute(2,0,1)
    bottom_point = points[..., 2:hd,   1:wd-1]
    top_point    = points[..., 0:hd-2, 1:wd-1]
    right_point  = points[..., 1:hd-1, 2:wd]
    left_point   = points[..., 1:hd-1, 0:wd-2]
    center_point = points[..., 1:hd-1, 1:wd-1]
    plane_normal = normal[..., 1:hd-1, 1:wd-1]
    plane_dist = (plane_normal * (top_point - center_point)).sum(dim=0) ** 2 + \
        (plane_normal * (left_point - center_point)).sum(dim=0).abs() ** 2 +\
        (plane_normal * (right_point - center_point)).sum(dim=0).abs() ** 2 +\
        (plane_normal * (bottom_point - center_point)).sum(dim=0).abs() ** 2
    return plane_dist

def depth_smoothness(depth):
    _, hd, wd = depth.shape 
    bottom_point = depth[..., 2:hd,   1:wd-1]
    right_point  = depth[..., 1:hd-1, 2:wd]
    center_point = depth[..., 1:hd-1, 1:wd-1]
    diff_img = (center_point - right_point) ** 2 + (center_point - bottom_point) ** 2
    return diff_img
    
    