import torch
import torch.nn.functional as F


def interpolate_embeddings(patch_embeddings, pixel_coords, patch_size, image_size):
    """
    Bilinearly interpolate a spatial feature map at arbitrary pixel coordinates.

    Normalizes so patch centers map exactly to [-1, 1]: patch centers lie at
    patch_size * i + (patch_size - 1) / 2, giving center = (W - 1) / 2
    and half-width = (W - patch_size) / 2, and analogously for H.

    Args:
        patch_embeddings: (H_feat * W_feat, D)
        pixel_coords: (N, 2) — (x, y) in [0, W-1] x [0, H-1].
        patch_size: spatial stride of the feature map in pixels.
        image_size: (H, W), must be divisible by patch_size.

    Returns:
        (N, D) — interpolated feature vector per query pixel.
    """
    assert patch_embeddings.ndim == 2
    assert pixel_coords.ndim == 2 and pixel_coords.shape[1] == 2

    H, W = image_size
    P, D = patch_embeddings.shape
    H_feat, W_feat = H // patch_size, W // patch_size
    device = patch_embeddings.device
    dtype = patch_embeddings.dtype

    assert H % patch_size == 0 and W % patch_size == 0
    assert P == H_feat * W_feat

    # we reshape embedding (P, D) -> spatial feature map (1, D, H_feat, W_feat)
    # Note: batch size is 1 since batch dimension is needed for grid_sample
    feature_map = patch_embeddings.reshape(1, H_feat, W_feat, D).permute(0, 3, 1, 2)

    # normalize pixel coords to [-1, 1] aligning corner patch centers to [-1, 1]
    # i.e. -1 -> P/2 and 1 -> H - P/2
    c = torch.tensor([[(W - 1) / 2, (H - 1) / 2]], dtype=dtype, device=device)
    norm = torch.tensor([[(W - patch_size) / 2, (H - patch_size) / 2]], dtype=dtype, device=device)
    grid = (pixel_coords - c) / norm

    # we reshape grid (N, 2) -> (1, N, 1, 2) expected by grid_sample
    grid = grid.view(1, -1, 1, 2)

    sampled = F.grid_sample(
        feature_map, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )
    return sampled.squeeze(0, -1).permute(1, 0)  # (1, D, N, 1) -> (N, D)


def interpolate_embeddings_conv(patch_embeddings, pixel_coords, stride, image_size):
    H, W = image_size
    H_feat, W_feat = H // stride, W // stride
    D = patch_embeddings.shape[1]
    device, dtype = patch_embeddings.device, patch_embeddings.dtype

    feat = patch_embeddings.reshape(1, H_feat, W_feat, D).permute(0, 3, 1, 2)

    grid = pixel_coords.clone().to(dtype=dtype, device=device)
    grid[:, 0] = (grid[:, 0] / stride + 0.5) / W_feat * 2 - 1
    grid[:, 1] = (grid[:, 1] / stride + 0.5) / H_feat * 2 - 1
    grid = grid.view(1, -1, 1, 2)

    sampled = F.grid_sample(feat, grid, mode="bilinear", align_corners=False, padding_mode="border")
    return sampled.squeeze(0, -1).permute(1, 0)


def midas_depth_transform(depth):
    """
    Convert depth map to MiDaS-style disparity.

    Closest pixels map to 1, farthest to 0. Invalid (zero) depth pixels
    are filled with the maximum valid depth before inversion so they
    appear as background (low disparity).

    Args:
        depth: (..., H, W) float tensor
    Returns:
        disparity: (..., H, W) tensor in [0, 1]
    """
    depth = depth.clone()
    valid_mask = depth > 0.0

    if not valid_mask.any():
        return torch.zeros_like(depth)

    depth[~valid_mask] = depth[valid_mask].max()
    depth = 1.0 / (depth + 1e-6)

    d_min = depth[valid_mask].min()
    d_max = depth[valid_mask].max()
    depth = (depth - d_min) / (d_max - d_min + 1e-6)
    depth = depth.clamp(0.0, 1.0)

    return depth


__all__ = ["interpolate_embeddings", "interpolate_embeddings_conv", "midas_depth_transform"]
