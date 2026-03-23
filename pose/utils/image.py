import numpy as np
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


def depth_to_patch_mask(depth, patch_size):
    """
    depth: (..., H, W)
    patch_size: int
    Returns:
        (..., N) where N = (H/P) * (W/P)
    """
    return patchify(depth.unsqueeze(-3), patch_size).gt(0).any(dim=-1)


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
    "depth_to_patch_mask",
    "get_crop_matrix",
    "get_pad_matrix",
    "get_resize_matrix",
]
