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


__all__ = ["interpolate_embeddings"]
