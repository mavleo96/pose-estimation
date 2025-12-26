import torch
from einops import einsum


def kabsch(
    src: torch.Tensor, tgt: torch.Tensor, w: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    src, tgt: (b, n, 3)
    w: optional (b, n)

    (b, n, dt) = (b, dt, ds) @ (b, n, ds) + (b, d)
    tgt = R @ src + t

    returns:
    R: (b, 3, 3)
    t: (b, 3)
    """
    assert src.shape == tgt.shape, "src and tgt must have the same shape"
    assert src.shape[2] == 3, "src and tgt must have 3 dimensions"
    assert src.shape[1] >= 3, "src and tgt must have at least 3 points"
    assert w is None or (w.shape == src.shape[:2]), "w must have the same shape as src"

    if w is None:
        w = torch.ones(src.shape[:2], device=src.device)

    # Normalize weights
    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
    w = w.unsqueeze(-1)

    # Compute centroids
    src_centroid = (w * src).sum(dim=1)
    tgt_centroid = (w * tgt).sum(dim=1)

    # Compute centered coordinates
    src_c = src - src_centroid.unsqueeze(1)
    tgt_c = tgt - tgt_centroid.unsqueeze(1)

    # Compute covariance matrix
    H = einsum(w * src_c, tgt_c, "b n ds, b n dt -> b ds dt")
    U, _, Vh = torch.linalg.svd(H)

    # Compute rotation matrix
    V = Vh.transpose(-1, -2)
    Ut = U.transpose(-1, -2)
    R = V @ Ut

    # Reflection fix
    det = torch.det(R)
    mask = det < 0
    if mask.any():
        V[mask, :, -1] *= -1
        R = V @ Ut

    # Compute translation
    t_est = tgt_centroid - einsum(R, src_centroid, "b dt ds, b ds -> b dt")
    return R, t_est


if __name__ == "__main__":
    from utils import rpy_to_rot

    R = rpy_to_rot(
        torch.tensor([torch.pi / 3]),
        torch.tensor([torch.pi / 3]),
        torch.tensor([torch.pi / 3]),
    )
    t = torch.arange(3, dtype=torch.float32).reshape(1, 3) / 10
    print(f"R: {R.round(decimals=2)}")
    print(f"t: {t.round(decimals=2)}")

    src = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.1, -0.2, 0.4]]).unsqueeze(
        0
    )
    tgt = einsum(R, src, "b dt ds, b n ds -> b n dt") + t
    w = torch.ones(1, 3)

    R_est, t_est = kabsch(src, tgt, w)
    print(f"R_est: {R_est.round(decimals=2)}")
    print(f"t_est: {t_est.round(decimals=2)}")

    # Error between R_est and R
    error_R = torch.norm(R_est - R, dim=(1, 2))
    print(f"error_R: {error_R.item():.3f}")

    # Error between t_est and t
    error_t = torch.norm(t_est - t, dim=1)
    print(f"error_t: {error_t.item():.3f}")
