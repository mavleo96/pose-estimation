import numpy as np
import torch
from einops import rearrange


def patchify(x, patch_size):
    """
    x: (..., C, H, W)
    patch_size: int
    Returns:
        (..., N, P * P * C) where N = (H/P) * (W/P)
    """
    return rearrange(
        x, "... c (ph p1) (pw p2) -> ... (ph pw) (p1 p2 c)", p1=patch_size, p2=patch_size
    )


def make_pixel_grid(h, w, device="cpu"):
    """
    h: int
    w: int
    device: torch.device or str
    Returns:
        (h, w, 2) in pixel coordinates
    """
    xs = torch.arange(w, device=device)
    ys = torch.arange(h, device=device)
    xs, ys = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack([xs, ys], dim=-1)


def sample_patch_centers(x, patch_size):
    """
    x: (..., H, W)
    patch_size: int
    Returns:
        (..., h, w) where h = H/P and w = W/P
    """
    return x[..., patch_size // 2 :: patch_size, patch_size // 2 :: patch_size]


def sample_pixels_from_masks(mask, N):
    """
    sample N pixels from the mask without replacement

    mask: (B, H, W)
    N: int
    Returns:
        (B, N, 2): (B, N, 2)
        (B, N): (B, N)
    """
    B, _, _ = mask.shape

    pixels = torch.zeros((B, N, 2), device=mask.device, dtype=torch.long)
    pixel_mask = torch.zeros((B, N), device=mask.device, dtype=torch.bool)

    for i in range(B):
        # randomly sample pixels from the mask (no replacement)
        inside_mask_indices = torch.argwhere(mask[i])
        num_available = inside_mask_indices.shape[0]

        if num_available == 0:
            # No pixels in mask
            continue
        elif num_available <= N:
            # Sample all available pixels without replacement
            indices = torch.randperm(num_available)
        else:
            # Sample N pixels without replacement
            indices = torch.randperm(num_available)[:N]
            num_available = N

        x, y = inside_mask_indices[indices, 1], inside_mask_indices[indices, 0]
        pixels[i, :num_available, 0] = x
        pixels[i, :num_available, 1] = y
        pixel_mask[i, :num_available] = True

    return pixels, pixel_mask


def apply_inverse_homography(pts2d, M):
    """
    find pts s.t. M * pts = pts2d

    pts2d: (..., N, 2)
    M:     (..., 3, 3)
    Returns:
           (..., N, 2)
    """
    ones = torch.ones_like(pts2d[..., :1], device=pts2d.device, dtype=pts2d.dtype)
    pts2d_h = torch.cat([pts2d, ones], dim=-1)  # (..., N, 3)
    result = torch.linalg.solve(M, pts2d_h.transpose(-2, -1))  # (..., 3, N)
    result = result.transpose(-2, -1)  # (..., N, 3)
    return result[..., :2] / result[..., 2:3]  # (..., N, 2)


def get_crop_matrix(bbox):
    """
    bbox: (x, y, w, h)
    Returns:
        M: (3, 3) transformation matrix mapping from original image to cropped image
    """
    x, y, _, _ = bbox
    return np.array(
        [
            [1, 0, -x],
            [0, 1, -y],
            [0, 0, 1],
        ],
        dtype=float,
    )


def get_pad_matrix(pad):
    """
    pad: (left, upper, right, lower)
    Returns:
        M: (3, 3) transformation matrix mapping from original image to padded image
    """
    left, upper, _, _ = pad
    return np.array(
        [
            [1, 0, left],
            [0, 1, upper],
            [0, 0, 1],
        ],
        dtype=float,
    )


def get_resize_matrix(src_size, dst_size):
    """
    src_size: (H, W)
    dst_size: (H, W)
    Returns:
        M: (3, 3) transformation matrix mapping from original image to resized image
    """
    sh, sw = src_size
    dh, dw = dst_size
    return np.array(
        [
            [dw / sw, 0, 0],
            [0, dh / sh, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )


__all__ = [
    "patchify",
    "make_pixel_grid",
    "sample_patch_centers",
    "apply_inverse_homography",
    "get_crop_matrix",
    "get_pad_matrix",
    "get_resize_matrix",
]
