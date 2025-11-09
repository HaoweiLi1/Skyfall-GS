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

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    """
    Compute PSNR in linear color space
    
    CRITICAL: Both images should be in linear space and clamped to [0, 1]
    """
    # Clamp to [0, 1] to avoid numerical issues
    img1 = torch.clamp(img1, 0.0, 1.0)
    img2 = torch.clamp(img2, 0.0, 1.0)
    
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    # Clamp MSE to avoid log(0)
    mse = torch.clamp(mse, min=1e-10)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
