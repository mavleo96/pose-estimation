import torch
from einops import einsum


class PoseEstimator:
    """
    Pose estimator using RANSAC algorithm.
    1. 3D-3D correspondences by feature matching
    2. RANSAC to estimate pose
    """

    def __init__(self, max_iters=1000, sample_size=3, inlier_thresh=0.05):
        self.max_iters = max_iters
        self.sample_size = sample_size
        self.inlier_thresh = inlier_thresh

    def estimate_pose(self, src_pts, src_feat, tar_pts, tar_feat):
        assert src_pts.shape[:-1] == src_feat.shape[:-1]
        assert tar_pts.shape[:-1] == tar_feat.shape[:-1]
        assert src_pts.ndim == tar_pts.ndim == 3
        assert src_pts.shape[0] == tar_pts.shape[0]
        assert src_pts.shape[-1] == tar_pts.shape[-1] == 3
        assert src_feat.shape[-1] == tar_feat.shape[-1]
        assert src_pts.shape[-2] <= tar_pts.shape[-2]

        b, n, _ = src_pts.shape
        _, m, _ = tar_pts.shape

        # get 3d-3d correspondences
        match_indices = self.correspondences_3d_3d(src_feat, tar_feat)
        match_indices = match_indices.unsqueeze(-1).expand(
            -1, -1, 3
        )  # b, n -> b, n, 1 -> b, n, 3
        corres_tar_pts = tar_pts.gather(dim=1, index=match_indices)

        # RANSAC to estimate pose
        best_num_inliers = torch.zeros(b, device=src_pts.device, dtype=torch.long)
        best_R = torch.zeros(b, 3, 3, device=src_pts.device)
        best_t = torch.zeros(b, 3, device=src_pts.device)
        for _ in range(self.max_iters):
            # sample points to estimate pose
            sample_indices = torch.randint(
                0, n, size=(b, self.sample_size), device=src_pts.device
            )
            src_pts_sample = src_pts.gather(
                dim=1, index=sample_indices.unsqueeze(-1).expand(-1, -1, 3)
            )
            tar_pts_sample = corres_tar_pts.gather(
                dim=1, index=sample_indices.unsqueeze(-1).expand(-1, -1, 3)
            )

            # estimate pose by kabsch algorithm
            R_est, t_est = self.kabsch(src_pts_sample, tar_pts_sample)

            # check inliners and update best pose
            est_pts = einsum(
                R_est, src_pts, "b d1 d2, b n d2 -> b n d1"
            ) + t_est.unsqueeze(1)
            errors = torch.norm(est_pts - corres_tar_pts, dim=2)
            inliers = errors < self.inlier_thresh
            num_inliers = inliers.sum(dim=1)
            better = num_inliers > best_num_inliers
            best_num_inliers[better] = num_inliers[better]
            best_R[better] = R_est[better]
            best_t[better] = t_est[better]

        # refit pose
        est_pts = einsum(
            best_R, src_pts, "b d1 d2, b n d2 -> b n d1"
        ) + best_t.unsqueeze(1)
        errors = torch.norm(est_pts - corres_tar_pts, dim=2)
        inliers = errors < self.inlier_thresh
        refit_mask = inliers.sum(dim=1) >= self.sample_size
        R_refit, t_refit = self.kabsch(src_pts[refit_mask], corres_tar_pts[refit_mask])
        best_R[refit_mask] = R_refit
        best_t[refit_mask] = t_refit

        return best_R, best_t

    @staticmethod
    def correspondences_3d_3d(src_feat, tar_feat):
        dist = torch.cdist(src_feat, tar_feat)
        match_indices = dist.argmin(-1)
        return match_indices

    @staticmethod
    def kabsch(src, tgt):
        """Pytorch implementation of Kabsch algorithm for rigid body registration.
        tgt = R @ src + t
        """
        assert src.shape == tgt.shape

        # compute centroids
        src_centroid = src.mean(dim=1, keepdim=True)
        tgt_centroid = tgt.mean(dim=1, keepdim=True)

        # compute cross-covariance matrix
        src_c = src - src_centroid
        tgt_c = tgt - tgt_centroid
        H = einsum(src_c, tgt_c, "b n ds, b n dt -> b ds dt")

        # compute rotation
        U, _, Vh = torch.linalg.svd(H)
        R = einsum(Vh, U, "b d dv, b du d -> b dv du")

        # fix reflection
        det = torch.det(R)
        mask = det < 0
        Vh[mask, :, -1] *= -1
        R = einsum(Vh, U, "b d dv, b du d -> b dv du")

        # compute translation
        t = tgt_centroid - einsum(R, src_centroid, "b d1 d2, b n d2 -> b n d1")  # n = 1
        return R, t.squeeze(1)


if __name__ == "__main__":
    pose_estimator = PoseEstimator()

    def feat_generator(pts):
        return torch.stack(
            [
                pts[:, :, 0] * pts[:, :, 1],
                pts[:, :, 0] * pts[:, :, 2],
                pts[:, :, 1] * pts[:, :, 2],
                torch.sin(pts[:, :, 0]),
                torch.cos(pts[:, :, 0]),
                torch.sin(pts[:, :, 1]),
                torch.cos(pts[:, :, 1]),
                torch.sin(pts[:, :, 2]),
                torch.cos(pts[:, :, 2]),
            ],
            dim=2,
        )

    src_pts = torch.randn(2, 100, 3)
    src_feat = feat_generator(src_pts)

    theta1 = torch.tensor(torch.pi / 4)
    theta2 = torch.tensor(torch.pi / 6)
    R = torch.tensor(
        [
            [
                torch.cos(theta1) * torch.cos(theta2),
                -torch.cos(theta1) * torch.sin(theta2),
                torch.sin(theta1),
            ],
            [
                torch.sin(theta1) * torch.cos(theta2),
                -torch.sin(theta1) * torch.sin(theta2),
                -torch.cos(theta1),
            ],
            [torch.sin(theta2), torch.cos(theta2), 0],
        ]
    ).expand(2, -1, -1)
    t = torch.tensor([2, -1, 3]).expand(2, -1)

    gt_tar_pts = torch.randn(2, 200, 3)
    tar_feat = feat_generator(gt_tar_pts)
    tar_pts = einsum(R, gt_tar_pts, "b d1 d2, b n d2 -> b n d1") + t.unsqueeze(1)

    R_est, t_est = pose_estimator.estimate_pose(src_pts, src_feat, tar_pts, tar_feat)
    print(f"R_est.shape, t_est.shape: {R_est.shape, t_est.shape}")
    print(f"R_est[0]: {R_est[0]}")
    print(f"t_est[0]: {t_est[0]}")
    print(f"R[0]: {R[0]}")
    print(f"t[0]: {t[0]}")
