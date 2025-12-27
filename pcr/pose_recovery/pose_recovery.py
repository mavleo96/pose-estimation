import torch
from einops import einsum
from .kabash import kabsch


class PoseEstimator:
    """
    Pose estimator using Kabsch + RANSAC algorithm.
    1. 3D-3D correspondences by feature matching; l2 distance
    2. Kabsch to estimate pose
    3. RANSAC to estimate inliers
    """

    def __init__(self, max_iters=1000, sample_size=3, inlier_thresh=100.0, tau=0.0):
        self.max_iters = max_iters
        self.sample_size = sample_size
        self.inlier_thresh = inlier_thresh
        if tau > 0.0:
            print(
                f"PoseEstimator: max_iters={max_iters}, sample_size={sample_size}, inlier_thresh={inlier_thresh}, tau={tau}, using soft correspondences"
            )
            self.tau = tau
        else:
            print(
                f"PoseEstimator: max_iters={max_iters}, sample_size={sample_size}, inlier_thresh={inlier_thresh}, using hard correspondences"
            )
            self.tau = None

    def estimate_pose(self, src_pts, src_feat, tar_pts, tar_feat):
        """
        Input:
            src_pts: (b, n, 3)
            src_feat: (b, n, d)
            tar_pts: (b, m, 3)
            tar_feat: (b, m, d)

        Score is the number of inliers / total number of points.

        Output:
            R: (b, 3, 3)
            t: (b, 3)
            score: (b,)
        """
        assert (
            src_pts.shape[0] == tar_pts.shape[0]
        ), f"src_pts and tar_pts batch size mismatch; got src {src_pts.shape[0]} and tar {tar_pts.shape[0]}"
        assert (
            src_pts.shape[:-1] == src_feat.shape[:-1]
        ), f"src pts and feat count mismatch; got src {src_pts.shape[:-1]} and feat {src_feat.shape[:-1]}"
        assert (
            tar_pts.shape[:-1] == tar_feat.shape[:-1]
        ), f"tar pts and feat count mismatch; got tar {tar_pts.shape[:-1]} and feat {tar_feat.shape[:-1]}"
        assert (
            src_pts.ndim == tar_pts.ndim == 3
        ), f"src_pts and tar_pts dimension mismatch; got src {src_pts.ndim} and tar {tar_pts.ndim}"
        assert (
            src_pts.shape[-1] == tar_pts.shape[-1] == 3
        ), f"src_pts and tar_pts coordinate mismatch; got src {src_pts.shape[-1]} and tar {tar_pts.shape[-1]}"
        assert (
            src_feat.shape[-1] == tar_feat.shape[-1]
        ), f"src_feat and tar_feat feature dimension mismatch; got src {src_feat.shape[-1]} and tar {tar_feat.shape[-1]}"
        assert (
            src_pts.shape[-2] <= tar_pts.shape[-2]
        ), f"src_pts must have at least as many points as tar_pts; got src {src_pts.shape[-2]} and tar {tar_pts.shape[-2]}"

        b, n, _ = src_pts.shape

        # 3D–3D correspondences
        if self.tau:
            w = self.soft_correspondences_3d_3d(
                src_feat, tar_feat, self.tau
            )  # b, ns, nt
            corres_tar_pts = w @ tar_pts
            w = w.sum(dim=-1)  # b, ns
        else:
            match_indices = self.hard_correspondences_3d_3d(src_feat, tar_feat)  # b, ns
            match_indices = match_indices.unsqueeze(-1).expand(
                -1, -1, 3
            )  # b, ns -> b, ns, 3
            corres_tar_pts = tar_pts.gather(dim=1, index=match_indices)
            w = torch.ones(b, n, device=src_pts.device, dtype=torch.float32)  # b, ns

        # Initialize best pose and inliers
        best_num_inliers = torch.zeros(b, device=src_pts.device, dtype=torch.long)
        best_R = torch.zeros(b, 3, 3, device=src_pts.device)
        best_t = torch.zeros(b, 3, device=src_pts.device)

        for _ in range(self.max_iters):
            sample_indices = torch.randint(
                n, size=(b, self.sample_size), device=src_pts.device
            )  # b, s
            sample_indices = sample_indices.unsqueeze(-1).expand(
                -1, -1, 3
            )  # b, s -> b, s, 3

            src_sample = src_pts.gather(1, sample_indices)
            tar_sample = corres_tar_pts.gather(1, sample_indices)
            w_sample = w.gather(1, sample_indices[:, :, 0])

            R_est, t_est = kabsch(src_sample, tar_sample, w_sample)

            est_pts = einsum(
                R_est, src_pts, "b dt ds, b n ds -> b n dt"
            ) + t_est.unsqueeze(1)

            errors = torch.norm(est_pts - corres_tar_pts, dim=2)
            inliers = errors < self.inlier_thresh
            num_inliers = inliers.sum(dim=1)

            better = num_inliers > best_num_inliers
            best_num_inliers[better] = num_inliers[better]
            best_R[better] = R_est[better]
            best_t[better] = t_est[better]

        # Refit using inliners
        est_pts = einsum(
            best_R, src_pts, "b dt ds, b n ds -> b n dt"
        ) + best_t.unsqueeze(1)
        errors = torch.norm(est_pts - corres_tar_pts, dim=2)
        inliers = errors < self.inlier_thresh
        refit_mask = inliers.sum(dim=1) >= self.sample_size
        if refit_mask.any():
            R_refit, t_refit = kabsch(
                src_pts[refit_mask], corres_tar_pts[refit_mask], w[refit_mask]
            )
            best_R[refit_mask] = R_refit
            best_t[refit_mask] = t_refit

        est_pts = einsum(
            best_R, src_pts, "b dt ds, b n ds -> b n dt"
        ) + best_t.unsqueeze(1)

        # Compute score by counting inliers
        errors = torch.norm(est_pts - corres_tar_pts, dim=2)
        score = (errors < self.inlier_thresh).sum(dim=1).float() / n
        print(
            f"errors: {errors.mean():.3f} {errors.std():.3f}, inliers: {inliers.sum(dim=1)}, inlier_thresh: {self.inlier_thresh}"
        )

        return best_R, best_t, score

    @staticmethod
    def hard_correspondences_3d_3d(src_feat, tar_feat):
        dist = torch.cdist(src_feat, tar_feat)  # b, ns, nt
        return dist.argmin(dim=-1)  # b, ns

    @staticmethod
    def soft_correspondences_3d_3d(src_feat, tar_feat, tau):
        dist = torch.cdist(src_feat, tar_feat)  # b, ns, nt
        w = torch.softmax(-dist / tau, dim=-1)  # b, ns, nt
        return w  # b, ns, nt


if __name__ == "__main__":
    import trimesh
    import numpy as np
    from .utils import rpy_to_rot
    from pcr.utils import transform_from_Rt
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    torch.manual_seed(42)
    np.random.seed(42)

    def feat_generator(pts):
        max_val = pts.max()
        return torch.stack(
            [
                pts[:, :, 0] * pts[:, :, 1] / max_val**2,
                pts[:, :, 0] * pts[:, :, 2] / max_val**2,
                pts[:, :, 1] * pts[:, :, 2] / max_val**2,
                torch.sin(pts[:, :, 0]),
                torch.cos(pts[:, :, 0]),
                torch.sin(pts[:, :, 1]),
                torch.cos(pts[:, :, 1]),
                torch.sin(pts[:, :, 2]),
                torch.cos(pts[:, :, 2]),
            ],
            dim=2,
        )

    R = rpy_to_rot(
        torch.tensor([4 * torch.pi / 5]),
        torch.tensor([torch.pi / 5]),
        torch.tensor([3 * torch.pi / 4]),
    )
    t = torch.tensor([120, -30, 600], dtype=torch.float32).unsqueeze(0)
    T = transform_from_Rt(R, t)

    print(f"R: {R.round(decimals=2)}")
    print(f"t: {t.round(decimals=2)}")

    path = "/workspace/pose-estimation/datasets/lmo/models/obj_000008.ply"
    mesh = trimesh.load(path)

    src_pts = trimesh.sample.sample_surface(mesh, 500)[0]
    src_pts = torch.tensor(src_pts, dtype=torch.float32).unsqueeze(0)
    src_feat = feat_generator(src_pts)

    # This is a hack to emulate the real data
    # We oversample tgt pts and filter for camera facing points
    gt_tar_pts_all = trimesh.sample.sample_surface(mesh, 3000)[0]
    gt_tar_pts_all = torch.tensor(gt_tar_pts_all, dtype=torch.float32).unsqueeze(0)
    tar_feat_all = feat_generator(gt_tar_pts_all)
    tar_pts_all = einsum(R, gt_tar_pts_all, "b d1 d2, b n d2 -> b n d1") + t.unsqueeze(1)

    # Filter for camera facing points
    def filter_camera_facing_points(mesh, T, pts):
        intersector = RayMeshIntersector(mesh.copy().apply_transform(T))
        locations, index_ray, _ = intersector.intersects_location(
            np.zeros_like(pts), pts, multiple_hits=False
        )
        visible_ray = np.full(pts.shape[0], False)
        hit_dist = np.linalg.norm(locations, axis=1)
        point_dist = np.linalg.norm(pts, axis=1)
        visible_ray[index_ray] = np.abs(hit_dist - point_dist[index_ray]) < 1e-2
        return visible_ray

    mask = filter_camera_facing_points(mesh, T.numpy()[0], tar_pts_all.numpy()[0])
    print(f"mask: {mask.sum()}")
    tar_pts = tar_pts_all[:, mask][:1000]
    tar_feat = tar_feat_all[:, mask][:1000]

    # Add noise to tgt pts
    tar_pts += torch.randn_like(tar_pts) * 50.0

    def geodesic_error(R_est, R_gt):
        R_diff = torch.matmul(R_est.T, R_gt)
        trace = torch.trace(R_diff)
        cos_theta = (trace - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1, 1)
        return torch.acos(cos_theta)

    pose_estimator = PoseEstimator(inlier_thresh=100.0)

    R_est, t_est, score = pose_estimator.estimate_pose(
        src_pts, src_feat, tar_pts, tar_feat
    )
    print(f"R_est: {R_est.round(decimals=2)}")
    print(f"t_est: {t_est.round(decimals=2)}")
    print(f"score: {score.item():.3f}")

    # Error between R_est and R
    geodesic_error_R = geodesic_error(R_est[0], R[0]) / torch.pi * 180
    print(f"geodesic_error_R: {geodesic_error_R.item():.3f} degrees")

    # Error between t_est and t
    error_t = torch.norm(t_est - t, dim=1) / torch.norm(t, dim=1)
    print(f"error_t: {error_t.item():.3f}")

    pose_estimator = PoseEstimator(inlier_thresh=100.0, tau=0.1)

    R_est, t_est, score = pose_estimator.estimate_pose(
        src_pts, src_feat, tar_pts, tar_feat
    )
    print(f"R_est: {R_est.round(decimals=2)}")
    print(f"t_est: {t_est.round(decimals=2)}")
    print(f"score: {score.item():.3f}")

    # Error between R_est and R
    geodesic_error_R = geodesic_error(R_est[0], R[0]) / torch.pi * 180
    print(f"geodesic_error_R: {geodesic_error_R.item():.3f} degrees")

    # Error between t_est and t
    error_t = torch.norm(t_est - t, dim=1) / torch.norm(t, dim=1)
    print(f"error_t: {error_t.item():.3f}")
