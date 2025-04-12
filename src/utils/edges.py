import torch

def compute_edges_batched(mask, height, width):
    """
    Computes edges between adjacent valid pixels (up, down, left, right)
    for a tensor of shape (..., h*w), with given height and width.

    Args:
        mask (Tensor): Binary mask of shape (..., h*w)
        height (int): Height of the spatial grid
        width (int): Width of the spatial grid

    Returns:
        List[Tensor]: A list of edge index tensors (2, num_edges) per (...,) grid
    """
    if mask.ndim < 2:
        raise ValueError(f"Expected mask of shape (..., h*w), but got {mask.shape}")

    *batch_dims, hw = mask.shape
    num_grids = int(torch.tensor(batch_dims).prod()) if batch_dims else 1

    if hw != height * width:
        raise ValueError(f"Expected last dim to be h*w = {height*width}, but got {hw}")

    device = mask.device
    mask_flat = mask.view(num_grids, height, width)

    all_edges = []

    idx_base = torch.arange(height * width, device=device).view(height, width)

    shifts = {
        'up':    ((slice(1, height), slice(0, width)), (slice(0, height - 1), slice(0, width))),
        'down':  ((slice(0, height - 1), slice(0, width)), (slice(1, height), slice(0, width))),
        'left':  ((slice(0, height), slice(1, width)), (slice(0, height), slice(0, width - 1))),
        'right': ((slice(0, height), slice(0, width - 1)), (slice(0, height), slice(1, width))),
    }

    for i in range(num_grids):
        curr_mask = mask_flat[i]
        edge_list = []

        for _, (src_slice, tgt_slice) in shifts.items():
            src_mask = curr_mask[src_slice]
            tgt_mask = curr_mask[tgt_slice]
            valid = (src_mask > 0) & (tgt_mask > 0)

            if valid.any():
                src_idx = idx_base[src_slice][valid].flatten()
                tgt_idx = idx_base[tgt_slice][valid].flatten()
                edges = torch.stack([src_idx, tgt_idx], dim=0)
                edge_list.append(edges)

        if edge_list:
            edges_tensor = torch.cat(edge_list, dim=1).contiguous()
        else:
            edges_tensor = torch.empty((2, 0), dtype=torch.long, device=device)

        all_edges.append(edges_tensor)

    return all_edges

if __name__ == "__main__":
    import numpy as np

    mask1 = torch.tensor(np.random.randint(0, 2, (4, 16 * 16)), dtype=torch.float)
    edges1 = compute_edges_batched(mask1, height=16, width=16)
    mask2 = torch.tensor(np.random.randint(0, 2, (4, 8, 16 * 16)), dtype=torch.float)
    edges2 = compute_edges_batched(mask2, height=16, width=16)

    print(f"input shape: {mask1.shape}; edges length: {len(edges1)}; edges shape: {edges1[0].shape}")
    print(f"input shape: {mask2.shape}; edges length: {len(edges2)}; edges shape: {edges2[0].shape}")
