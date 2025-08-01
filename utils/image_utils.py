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
from scipy import ndimage

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(input_tensor, structure=None):
    input_tensor = torch.where(input_tensor.bool(), torch.ones_like(input_tensor).float(), torch.zeros_like(input_tensor).float())
    if structure is None:
        structure = torch.from_numpy(ndimage.generate_binary_structure(2, 1)[None,None,...]).to(input_tensor).float()
        #structure = torch.ones((1, 1, 3, 3), dtype=input_tensor.dtype, device=input_tensor.device)
    eroded = F.conv2d(input_tensor.unsqueeze(0).unsqueeze(0), structure, padding=1)[0][0]
    return (eroded == 5.).bool()
