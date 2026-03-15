import torch
from einops import einsum


class KabschPoseEstimator:
    """
    Batched pose estimator using Kabsch + RANSAC.

    RANSAC draws minimal samples (3 point pairs) to hypothesize a rigid
    transform, scores each hypothesis by counting inliers across all
    correspondences, retains the best hypothesis per batch item, then
    refits on the full inlier set using weighted Kabsch.

    Args:
        max_iters (int): Number of RANSAC iterations. Default: 1000.
        inlier_thresh (float): Residual threshold (same units as input
            points) below which a correspondence is considered an inlier.
            For BOP-format datasets where points are in millimetres a
            value in the range 5–15 mm is typical; the default of 10.0
            is a reasonable starting point. Default: 10.0.
    """

    def __init__(self, max_iters=1000, inlier_thresh=10.0) -> None:
        self.max_iters = max_iters
        self.inlier_thresh = inlier_thresh
        self.sample_size = 3

    def estimate_pose(self, src_pts, tgt_pts, mask=None):
        """
        Estimate the rigid transformation from src to tgt via Kabsch+RANSAC.

        Args:
            src_pts (torch.Tensor): Source 3-D points, shape (B, N, 3).
            tgt_pts (torch.Tensor): Target 3-D points, shape (B, N, 3).
                Correspondence is positional: src_pts[:, i, :] pairs with
                tgt_pts[:, i, :].
            mask (torch.Tensor | None): Boolean validity mask, shape (B, N).
                True  = valid correspondence included in all computations.
                False = padding slot excluded from sampling, inlier
                        counting, covariance estimation, and scoring.
                If None, all N positions are treated as valid.

        Returns:
            best_R (torch.Tensor): Rotation matrices, shape (B, 3, 3).
                Identity for items with fewer than sample_size valid
                correspondences.
            best_t (torch.Tensor): Translation vectors, shape (B, 3).
                Zero for items with fewer than sample_size valid
                correspondences.
            score (torch.Tensor): Inlier fraction in [0, 1], shape (B,).
                Denominator is valid_count per item, not padded N.
                Zero for degenerate items.
        """
        assert src_pts.shape == tgt_pts.shape, "src_pts and tgt_pts must have the same shape"
        assert mask is None or mask.shape == src_pts.shape[:2], (
            "mask must have the same shape as src_pts"
        )

        B, N, _ = src_pts.shape
        device = src_pts.device

        if mask is None:
            mask = torch.ones(B, N, device=device, dtype=torch.bool)

        valid_count = mask.sum(dim=1)
        valid_enough = valid_count >= self.sample_size

        # Initialize best pose and inliers
        best_R = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
        best_t = torch.zeros(B, 3, device=device)
        best_num_inliers = torch.zeros(B, device=device, dtype=torch.long)

        if not valid_enough.any():
            return best_R, best_t, torch.zeros(B, device=device)

        # RANSAC loop
        sample_probs = mask.float()

        for _ in range(self.max_iters):
            sample_idx = torch.multinomial(sample_probs, self.sample_size, replacement=False)
            sample_idx = sample_idx.unsqueeze(-1).expand(-1, -1, 3)

            src_sample = src_pts.gather(1, sample_idx)
            tgt_sample = tgt_pts.gather(1, sample_idx)

            R_hyp, t_hyp = self.kabsch(src_sample, tgt_sample)

            est_pts = einsum(R_hyp, src_pts, "b dt ds, b n ds -> b n dt") + t_hyp.unsqueeze(1)
            errors = torch.norm(est_pts - tgt_pts, dim=2)

            inlier_mask = (errors < self.inlier_thresh) & mask
            num_inliers = inlier_mask.sum(dim=1)

            better = (num_inliers > best_num_inliers) & valid_enough
            if better.any():
                ids = torch.where(better)[0]
                best_num_inliers[ids] = num_inliers[ids]
                best_R[ids] = R_hyp[ids]
                best_t[ids] = t_hyp[ids]

        # Refit using inliners
        est_pts = einsum(best_R, src_pts, "b dt ds, b n ds -> b n dt") + best_t.unsqueeze(1)
        errors = torch.norm(est_pts - tgt_pts, dim=2)
        inlier_mask = (errors < self.inlier_thresh) & mask
        refit_ok = (inlier_mask.sum(dim=1) >= self.sample_size) & valid_enough

        if refit_ok.any():
            w = inlier_mask.float()  # (B, N)
            R_refit, t_refit = self.kabsch(src_pts, tgt_pts, w)
            ids = torch.where(refit_ok)[0]
            best_R[ids] = R_refit[ids]
            best_t[ids] = t_refit[ids]

        # Compute score by counting inliers
        est_pts = einsum(best_R, src_pts, "b dt ds, b n ds -> b n dt") + best_t.unsqueeze(1)
        errors = torch.norm(est_pts - tgt_pts, dim=2)
        inlier_mask = (errors < self.inlier_thresh) & mask
        score = inlier_mask.sum(dim=1).float() / valid_count.float().clamp(min=1.0)
        score[~valid_enough] = 0.0

        return best_R, best_t, score

    @staticmethod
    def kabsch(src, tgt, w=None):
        """
        Weighted Kabsch algorithm: optimal rigid transform via SVD.

        Minimises  sum_i  w_i * || R @ src_i + t - tgt_i ||^2

        Convention:  tgt ~= R @ src + t

        Args:
            src (torch.Tensor): Source points, shape (B, N, 3).
            tgt (torch.Tensor): Target points, shape (B, N, 3).
            w (torch.Tensor | None): Non-negative per-point weights,
                shape (B, N). Normalised internally so only relative
                magnitudes matter. w_i = 0 excludes point i from centroid
                and covariance — used to mask outliers and padding slots
                during the refit step without ragged tensors.
                Uniform weights if None.

        Returns:
            R (torch.Tensor): Rotation matrices,   shape (B, 3, 3).
            t (torch.Tensor): Translation vectors, shape (B, 3).

        Notes:
            - Reflection ambiguity (det R = -1) is resolved by negating
              the last column of V before forming R = V @ U^T.
            - Weight normalisation includes eps=1e-8 to safely handle
              degenerate all-zero weight vectors.
        """
        assert src.shape == tgt.shape, "src and tgt must have the same shape"
        assert src.shape[2] == 3, "src and tgt must have 3 dimensions"
        assert src.shape[1] >= 3, "src and tgt must have at least 3 points"
        assert w is None or w.shape == src.shape[:2], "w must have the same shape as src"

        if w is None:
            w = torch.ones(src.shape[:2], device=src.device, dtype=src.dtype)

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
        V = Vh.mT
        Ut = U.mT
        R = V @ Ut

        # Reflection fix: negate last column of V when det(R) < 0
        neg_det = torch.linalg.det(R) < 0
        if neg_det.any():
            V[neg_det, :, -1] *= -1
            R = V @ Ut

        # Compute translation
        t = tgt_centroid - einsum(R, src_centroid, "b dt ds, b ds -> b dt")
        return R, t


if __name__ == "__main__":
    import trimesh
    import numpy as np
    from ..utils import euler_to_rotation_matrix
    from _configs.constants import DATA_DIR
    from pathlib import Path

    torch.manual_seed(42)
    np.random.seed(42)

    R = euler_to_rotation_matrix(
        torch.tensor([4 * torch.pi / 5]),
        torch.tensor([torch.pi / 5]),
        torch.tensor([3 * torch.pi / 4]),
    )
    t = torch.tensor([120, -30, 600], dtype=torch.float32).unsqueeze(0)
    T = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).expand(R.shape[0], -1, -1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0

    print(f"R: {R.round(decimals=2)}")
    print(f"t: {t.round(decimals=2)}")

    model_path = Path(DATA_DIR) / "lmo" / "models" / "obj_000008.ply"
    mesh = trimesh.load(model_path)

    src_pts = trimesh.sample.sample_surface(mesh, 500)[0]
    src_pts = torch.tensor(src_pts, dtype=torch.float32).unsqueeze(0)
    tgt_pts = einsum(R, src_pts, "b d1 d2, b n d2 -> b n d1") + t.unsqueeze(1)
    mask = torch.zeros(*src_pts.shape[:2], device=src_pts.device, dtype=torch.bool)
    mask[:, :400] = True

    # Add noise to tgt pts
    tgt_pts += torch.randn_like(tgt_pts) * 50.0
    # Add some outliers
    tgt_pts[:, :50] = (
        torch.randn(50, 3, device=src_pts.device, dtype=src_pts.dtype) * src_pts.std()
        + src_pts.mean()
    )

    def geodesic_error(R_est, R_gt):
        R_diff = torch.matmul(R_est.T, R_gt)
        trace = torch.trace(R_diff)
        cos_theta = (trace - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1, 1)
        return torch.acos(cos_theta)

    pose_estimator = KabschPoseEstimator(inlier_thresh=100.0)

    R_est, t_est, score = pose_estimator.estimate_pose(src_pts, tgt_pts, mask)
    print(f"R_est: {R_est.round(decimals=2)}")
    print(f"t_est: {t_est.round(decimals=2)}")
    print(f"score: {score.item():.3f}")

    # Error between R_est and R
    geodesic_error_R = geodesic_error(R_est[0], R[0]) / torch.pi * 180
    print(f"geodesic_error_R: {geodesic_error_R.item():.3f} degrees")

    # Error between t_est and t
    error_t = torch.norm(t_est - t, dim=1) / torch.norm(t, dim=1)
    print(f"error_t: {error_t.item():.3f}")
