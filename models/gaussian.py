import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import to_pixel_samples
import time
from torchvision.utils import save_image
from itertools import product

from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def generate_meshgrid(height, width):
    """
    Generate a meshgrid of coordinates for a given image dimensions.
    Args:
        height (int): Height of the image.
        width (int): Width of the image.
    Returns:
        torch.Tensor: A tensor of shape [height * width, 2] containing the (x, y) coordinates for each pixel in the image.
    """
    # Generate all pixel coordinates for the given image dimensions
    y_coords, x_coords = torch.arange(0, height), torch.arange(0, width)
    # Create a grid of coordinates
    yy, xx = torch.meshgrid(y_coords, x_coords)
    # Flatten and stack the coordinates to obtain a list of (x, y) pairs
    all_coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return all_coords


def fetching_features_from_tensor(image_tensor, input_coords):
    """
    Extracts pixel values from a tensor of images at specified coordinate locations.
    Args:
        image_tensor (torch.Tensor): A 4D tensor of shape [batch, channel, height, width] representing a batch of images.
        input_coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the (x, y) coordinates at which to extract pixel values.
    Returns:
        color_values (torch.Tensor): A 3D tensor of shape [batch, N, channel] containing the pixel values at the specified coordinates.
        coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the normalized coordinates in the range [-1, 1].
    """
    # Normalize pixel coordinates to [-1, 1] range
    input_coords = input_coords.to(image_tensor.device)
    coords = input_coords / torch.tensor([image_tensor.shape[-2], image_tensor.shape[-1]],
                                         device=image_tensor.device).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=image_tensor.device).float()
    coords = (center_coords_normalized - coords) * 2.0

    # Fetching the colour of the pixels in each coordinates
    batch_size = image_tensor.shape[0]
    input_coords_expanded = input_coords.unsqueeze(0).expand(batch_size, -1, -1)

    y_coords = input_coords_expanded[..., 0].long()
    x_coords = input_coords_expanded[..., 1].long()
    batch_indices = torch.arange(batch_size).view(-1, 1).to(input_coords.device)

    color_values = image_tensor[batch_indices, :, x_coords, y_coords]

    return color_values, coords


def scale_to_range(tensor, min_value, max_value):
    min_tensor = torch.min(tensor)
    max_tensor = torch.max(tensor)
    scaled_tensor = (tensor - min_tensor) / (max_tensor - min_tensor)  
    return scaled_tensor * (max_value - min_value) + min_value


def get_coord(width, height):
    x_coords = torch.arange(width)
    y_coords = torch.arange(height)

    # Generate coordinate grid using torch.meshgrid
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')

    # Map coordinates to the range of -1 to 1
    x_grid = 2 * (x_grid / (width)) - 1 #+ 1/width
    y_grid = 2 * (y_grid / (height)) - 1 #+ 1/height

    # Stack the x and y coordinates to form the final coordinate tensor
    coordinates = torch.stack((y_grid, x_grid), dim=-1).reshape(-1, 2)
    
    return coordinates


@register('continuous-gaussian')
class ContinuousGaussian(nn.Module):
    """A module that applies 2D Gaussian splatting to input features."""

    def __init__(self, encoder_spec, cnn_spec, fc_spec, **kwargs):
        """
        Initialize the ContinuousGaussian module.
        
        Args:
            encoder_spec (dict): Specifications for the encoder.
            cnn_spec (dict): Specifications for the CNN layers.
            fc_spec (dict): Specifications for the fully connected layers.
            kwargs: Additional arguments.
        """
        super(ContinuousGaussian, self).__init__()
        
        # Create the encoder module based on the given specifications
        self.encoder = models.make(encoder_spec)
        
        # Initialize placeholders for various attributes
        self.feat = None  # Low-resolution (LR) features
        self.inp = None  # Input image
        self.feat_coord = None  # Feature coordinates
        self.init_num_points = None
        self.H, self.W = None, None  # Image height and width
        self.BLOCK_H, self.BLOCK_W = 16, 16  # Block size for tiling

        # Define additional convolutional and activation layers
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # Convolutional layer
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # Leaky ReLU activation
        self.ps = nn.PixelUnshuffle(2)  # Pixel unshuffle with a scaling factor of 2

        # Define an MLP for vector generation for gaussian dict
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 3, 'out_dim': 512, 'hidden_list': [256, 512, 512, 512]}}
        self.mlp_vector = models.make(mlp_spec)
        
        # Define an MLP for color prediction
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 256, 'out_dim': 3, 'hidden_list': [512, 1024, 256, 128, 64]}}
        self.mlp = models.make(mlp_spec)
        
        # Define an MLP for offset prediction
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 256, 'out_dim': 2, 'hidden_list': [512, 1024, 256, 128, 64]}}
        self.mlp_offset = models.make(mlp_spec)
        
        # Initialize pre-defined Gaussian convariance parameter dictionaries
        cho1 = torch.tensor([0, 0.41, 0.62, 0.98, 1.13, 1.29, 1.64, 1.85, 2.36])
        cho2 = torch.tensor([-0.86, -0.36, -0.16, 0.19, 0.34, 0.49, 0.84, 1.04, 1.54])
        cho3 = torch.tensor([0, 0.33, 0.53, 0.88, 1.03, 1.18, 1.53, 1.73, 2.23])
        self.gau_dict = torch.tensor(list(product(cho1, cho2, cho3)))
        self.gau_dict = torch.cat((self.gau_dict, torch.zeros(1, 3)), dim=0)  # Add zeros

        # Define SGS covariance head
        self.num_sgs_kernels = 100  # Number of kernels in the bank (this can be tuned)
        # Learnable bank of (sigma_x, sigma_y, rho, opacity)
        sigma_x, sigma_y = torch.meshgrid(
            torch.linspace(0.2, 3.0, 10), 
            torch.linspace(0.2, 3.0, 10)
        )
        sigma_x = sigma_x.reshape(-1)[:self.num_sgs_kernels]
        sigma_y = sigma_y.reshape(-1)[:self.num_sgs_kernels]

        self.sgs_sigma_x = nn.Parameter(sigma_x)                        
        self.sgs_sigma_y = nn.Parameter(sigma_y)                        
        self.sgs_rho = nn.Parameter(torch.zeros(self.num_sgs_kernels))  
        self.sgs_opacity = nn.Parameter(torch.sigmoid(
            torch.ones(self.num_sgs_kernels)
        ))  

        # kernel-selection logits head (SGS classifier over kernels)
        # input channels = 256 (same as feature channels)
        self.sgs_logits_head = nn.Conv2d(256, self.num_sgs_kernels, kernel_size=1)

        # gating head alpha \in (0,1) to mix DDCW vs SGS
        self.alpha_head = nn.Conv2d(256, 1, kernel_size=1)

        self.last_size = (self.H, self.W)
        self.background = torch.ones(3).cuda()  # Default background color

    def gen_feat(self, inp):
        """
        Generate features from the input image using the encoder.
        
        Args:
            inp (torch.Tensor): Input image.
        
        Returns:
            torch.Tensor: Extracted low-resolution features.
        """
        self.inp = inp  # Store the input image
        feat = self.encoder(inp)  # Encode the input image
        self.feat = self.ps(feat)  # Apply pixel unshuffle to the encoded features
        return self.feat

    def sgs_gaussian_params(self, logits):
        """
        ContinuousSR-friendly SGS: map logits over a kernel bank to per-location
        Cholesky parameters (l1, l2, l3) and opacity.

        logits: [B, K_sgs, H_feat, W_feat]
        returns:
            l_chol: [B, H_feat*W_feat, 3]   # (l1, l2, l3)
            opacity: [B, H_feat*W_feat, 1]
        """
        B, K, H, W = logits.shape
        # convert to probabilities over kernels
        weights = torch.softmax(logits, dim=1)        # [B, K, H, W]
        weights = weights.permute(0, 2, 3, 1)         # [B, H, W, K]

        device = logits.device
        sigma_x_bank = self.sgs_sigma_x.to(device).view(1, 1, 1, K)      # [1,1,1,K]
        sigma_y_bank = self.sgs_sigma_y.to(device).view(1, 1, 1, K)
        rho_bank = self.sgs_rho.to(device).view(1, 1, 1, K)
        opacity_bank = self.sgs_opacity.to(device).view(1, 1, 1, K)

        # weighted parameters per spatial location
        sigma_x = (weights * sigma_x_bank).sum(dim=-1)       # [B,H,W]
        sigma_y = (weights * sigma_y_bank).sum(dim=-1)       # [B,H,W]
        rho = (weights * rho_bank).sum(dim=-1).clamp(-0.99, 0.99)   # [B,H,W]
        opacity = (weights * opacity_bank).sum(dim=-1)       # [B,H,W]

        # convert (sigma_x, sigma_y, rho) -> Cholesky L parameters
        # Σ = [[σx^2, ρσxσy], [ρσxσy, σy^2]]
        # L = [[l1, 0],
        #      [l2, l3]]  with:
        #   l1 = σx
        #   l2 = ρ σy
        #   l3 = σy * sqrt(1 - ρ^2)
        eps = 1e-6
        l1 = sigma_x
        l2 = rho * sigma_y
        l3 = sigma_y * torch.sqrt(1.0 - rho ** 2 + eps)

        # flatten to [B, H*W, 1]
        num_kernels = H * W
        l1 = l1.view(B, num_kernels, 1)
        l2 = l2.view(B, num_kernels, 1)
        l3 = l3.view(B, num_kernels, 1)
        opacity = opacity.view(B, num_kernels, 1)

        l_chol = torch.cat([l1, l2, l3], dim=-1)   # [B, num_kernels, 3]

        return l_chol, opacity

    def query_output(self, inp, scale):
        """
        Generate the high-resolution image output for a given scale.
        
        Args:
            inp (torch.Tensor): Input image.
            scale (torch.Tensor): Scaling factor.
        
        Returns:
            torch.Tensor: High-resolution output image.
        """
        feat = self.feat
        device = feat.device

        # Process the scaling factors
        if scale.shape == (1, 2):  # Handle cases with two scaling factors
            scale1 = float(scale[0, 0])
            scale2 = float(scale[0, 1])
        else:  # Handle uniform scaling
            scale1 = float(scale[0])
            scale2 = float(scale[0])

        # Compute dimensions of the low-resolution and high-resolution images
        lr_h = self.inp.shape[-2]  # Low-resolution height
        lr_w = self.inp.shape[-1]  # Low-resolution width
        H = round(int(lr_h) * scale1)  # High-resolution height
        W = round(int(lr_w) * scale2)  # High-resolution width

        # Determine the number of tiles for rasterization
        self.tile_bounds = (
            (W + self.BLOCK_W - 1) // self.BLOCK_W,
            (H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )

        window_size = 1  # Window size for Gaussian position adjustments
        pred = []  # List to store predictions

        # -------------------------------
        # Shapes and Gaussian count
        # -------------------------------
        bs, c_feat, h_feat, w_feat = feat.shape    # [B, 256, H_f, W_f]
        num_gauss = (lr_h * 2) * (lr_w * 2)        # one Gaussian per HR 2× grid location

        # -------------------------------
        # 1) Upsample features to 2× LR grid
        # -------------------------------
        # feat_up: [B, 256, 2*lr_h, 2*lr_w]
        feat_up = F.interpolate(
            feat, size=(lr_h * 2, lr_w * 2),
            mode='bilinear', align_corners=False
        )

        # Flatten to per-Gaussian 256-dim vectors
        # feat_flat: [B * num_gauss, 256]
        feat_flat = feat_up.permute(0, 2, 3, 1).reshape(bs * num_gauss, c_feat)

        # -------------------------------
        # 2) CGM – color prediction
        # -------------------------------
        color = self.mlp(feat_flat)                # [B*num_gauss, 3]
        color = color.view(bs, num_gauss, 3)       # [B, num_gauss, 3]

        # -------------------------------
        # 3) DDCW – global covariance (Deep Gaussian Prior)
        # -------------------------------
        para_c = self.leaky_relu(feat)             # [B, 256, H_f, W_f]
        feat_conv = self.conv1(para_c)             # [B, 512, H_f, W_f]

        # Upsample conv features to 2× LR grid
        # feat_conv_up: [B, 512, 2*lr_h, 2*lr_w]
        feat_conv_up = F.interpolate(
            feat_conv, size=(lr_h * 2, lr_w * 2),
            mode='bilinear', align_corners=False
        )

        # Flatten to [B*num_gauss, 512]
        feat_conv_flat = feat_conv_up.permute(0, 2, 3, 1).reshape(bs * num_gauss, 512)

        gau_dict_device = self.gau_dict.to(device)         # [N_dict, 3]
        vector = self.mlp_vector(gau_dict_device)          # [N_dict, 512]

        # para_weights: [B*num_gauss, N_dict]
        para_weights = feat_conv_flat @ vector.t()
        para_weights = torch.softmax(para_weights, dim=-1)

        # para_ddcw: [B*num_gauss, 3] -> [B, num_gauss, 3]
        para_ddcw = para_weights @ gau_dict_device         # [B*num_gauss, 3]
        para_ddcw = para_ddcw.view(bs, num_gauss, 3)

        # -------------------------------
        # 4) SGS – local covariance from kernel bank
        # -------------------------------
        # logits over SGS kernel bank from LR feature map
        sgs_logits = self.sgs_logits_head(feat)            # [B, K_sgs, H_f, W_f]

        # Upsample logits to 2× LR grid so we have one kernel per Gaussian
        sgs_logits = F.interpolate(
            sgs_logits, size=(lr_h * 2, lr_w * 2),
            mode='bilinear', align_corners=False
        )                                                   # [B, K_sgs, 2*lr_h, 2*lr_w]

        # para_sgs: [B, num_gauss, 3], opacity_sgs: [B, num_gauss, 1]
        para_sgs, opacity_sgs = self.sgs_gaussian_params(sgs_logits)

        # -------------------------------
        # 5) α-gating – mix DDCW and SGS per location
        # -------------------------------
        alpha_logit = self.alpha_head(feat)                 # [B,1,H_f,W_f]
        alpha_logit = F.interpolate(
            alpha_logit, size=(lr_h * 2, lr_w * 2),
            mode='bilinear', align_corners=False
        )                                                   # [B,1,2*lr_h,2*lr_w]

        alpha = torch.sigmoid(alpha_logit).view(bs, num_gauss, 1)  # [B, num_gauss, 1]

        # final Cholesky params and opacity
        para    = alpha * para_ddcw + (1.0 - alpha) * para_sgs     # [B, num_gauss, 3]
        opacity = alpha * 1.0      + (1.0 - alpha) * opacity_sgs   # [B, num_gauss, 1]

        # -------------------------------
        # 6) APD – position offsets from 256-dim features
        # -------------------------------
        offset = self.mlp_offset(feat_flat)                # [B*num_gauss, 2]
        offset = torch.tanh(offset).view(bs, num_gauss, 2) # [B, num_gauss, 2]

        # -------------------------------
        # 7) Base coordinates on 2× LR grid
        # -------------------------------
        base_xyz = get_coord(lr_h * 2, lr_w * 2).to(device)  # [num_gauss, 2]

        # -------------------------------
        # 8) Per-batch rendering
        # -------------------------------
        for i in range(bs):
            offset_i  = offset[i]         # [num_gauss, 2]
            color_i   = color[i]          # [num_gauss, 3]
            para_i    = para[i]           # [num_gauss, 3]
            opacity_i = opacity[i]        # [num_gauss, 1]

            get_xyz = base_xyz

            # Adjust coordinates using offsets
            xyz1 = get_xyz[:, 0:1] + 2 * window_size * offset_i[:, 0:1] / lr_w - 1.0 / W
            xyz2 = get_xyz[:, 1:2] + 2 * window_size * offset_i[:, 1:2] / lr_h - 1.0 / H
            get_xyz_i = torch.cat((xyz1, xyz2), dim=1)      # [num_gauss, 2]

            # Adjust Gaussian parameters
            weighted_cholesky = para_i / 4.0                # [num_gauss, 3]
            weighted_cholesky[:, 0] *= scale2
            weighted_cholesky[:, 1] *= scale1
            weighted_cholesky[:, 2] *= scale1

            # Perform Gaussian projection and rasterization
            xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
                get_xyz_i, weighted_cholesky, H, W, self.tile_bounds
            )

            out_img = rasterize_gaussians_sum(
                xys, depths, radii, conics, num_tiles_hit,
                color_i, opacity_i,
                H, W, self.BLOCK_H, self.BLOCK_W,
                background=self.background, return_alpha=False
            )
            out_img = out_img.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
            pred.append(out_img)

        # Combine outputs for the batch
        out_img = torch.cat(pred, dim=0)   # [B,3,H,W]
        return out_img


    def forward(self, inp, scale):
        """
        Forward pass for the ContinuousGaussian module.
        
        Args:
            inp (torch.Tensor): Input image.
            scale (torch.Tensor): Scaling factor.
        
        Returns:
            torch.Tensor: High-resolution output image.
        """
        self.gen_feat(inp)  # Generate low-resolution features
        image = self.query_output(inp, scale)  # Generate high-resolution output
        return image
