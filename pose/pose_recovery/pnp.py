import cv2
import numpy as np
import torch


class PnPPoseEstimator:
    """
    Batched PnP pose estimator wrapping cv2.solvePnPRansac (SQPnP solver)
    with optional Levenberg-Marquardt refit on inliers.

    cv2.solvePnPRansac handles outlier rejection internally, so no outer
    RANSAC loop is needed. The estimator loops over batch items (cv2 has
    no batched interface) and collects results back into tensors.

    Args:
        inlier_thresh (float): Reprojection error threshold in pixels
            below which a correspondence is considered an inlier.
            Default: 8.0.
        confidence (float): RANSAC confidence level in [0, 1].
            Default: 0.99.
        max_iters (int): Maximum RANSAC iterations passed to cv2.
            Default: 1000.
        refine (bool): If True, run cv2.solvePnPRefineLM on the inlier
            set after RANSAC. Improves accuracy at modest cost.
            If refinement fails (e.g. degenerate inlier geometry), the
            RANSAC result is kept silently. Default: True.
    """

    def __init__(
        self,
        inlier_thresh=8.0,
        confidence=0.99,
        max_iters=1000,
        refine=True,
    ):
        self.inlier_thresh = inlier_thresh
        self.confidence = confidence
        self.max_iters = max_iters
        self.refine = refine

    def __repr__(self):
        return (
            f"PnPPoseEstimator(inlier_thresh={self.inlier_thresh}, confidence={self.confidence},"
            f" max_iters={self.max_iters}, refine={self.refine})"
        )

    def estimate_pose(self, pts3d, pts2d, K, mask=None):
        """
        Estimate camera pose via cv2.solvePnPRansac (SQPnP).

        Args:
            pts3d (torch.Tensor): 3-D world points, shape (B, N, 3).
            pts2d (torch.Tensor): 2-D image points in pixels, shape (B, N, 2).
                Correspondence is positional: pts3d[:, i, :] projects to
                pts2d[:, i, :].
            K (torch.Tensor): Camera intrinsic matrices, shape (B, 3, 3).
            mask (torch.Tensor | None): Boolean validity mask, shape (B, N).
                True  = valid correspondence included in all computations.
                False = padding slot excluded from PnP and scoring.
                If None, all N positions are treated as valid.

        Returns:
            best_R (torch.Tensor): Rotation matrices, shape (B, 3, 3).
                Identity for failed or degenerate items.
            best_t (torch.Tensor): Translation vectors, shape (B, 3).
                Zero for failed or degenerate items.
            score (torch.Tensor): Inlier fraction in [0, 1], shape (B,).
                Numerator is inlier count reported by cv2 (or valid_count
                if cv2 does not report inliers). Denominator is valid_count
                per item, not padded N. Zero for failed or degenerate items.

        Notes:
            - Convention: lambda * [u, v, 1]^T = K @ (R @ X + t).
            - Assumes undistorted image points. Pass distortion coefficients
              via a subclass or modify distCoeffs if needed.
            - Items with fewer than min_points (4) valid correspondences
              receive identity R, zero t, and score 0.
            - All cv2 operations run on CPU in float64 for numerical
              stability; outputs are cast back to the input dtype and device.
            - If solvePnPRefineLM raises (degenerate inlier geometry),
              the RANSAC result is silently kept.
        """
        assert pts3d.shape[:2] == pts2d.shape[:2], "pts3d and pts2d must have the same (B, N)"
        assert pts3d.shape[2] == 3, "pts3d must be (B, N, 3)"
        assert pts2d.shape[2] == 2, "pts2d must be (B, N, 2)"
        assert K.shape == (pts3d.shape[0], 3, 3), "K must be (B, 3, 3)"
        assert mask is None or mask.shape == pts3d.shape[:2], "mask must be (B, N)"

        B, N, _ = pts3d.shape
        device = pts3d.device
        dtype = pts3d.dtype

        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)

        valid_count = mask.sum(dim=1)  # (B,)  int tensor
        valid_enough = valid_count >= 4  # (B,)  bool tensor

        # Initialize best pose and inliers
        best_R = torch.eye(3, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        best_t = torch.zeros(B, 3, dtype=dtype)
        score = torch.zeros(B, dtype=dtype)

        if not valid_enough.any():
            return best_R.to(device), best_t.to(device), score.to(device)

        # Move to CPU numpy — cv2 is CPU-only and requires float64.
        pts3d_np = pts3d.detach().cpu().numpy().astype(np.float64)
        pts2d_np = pts2d.detach().cpu().numpy().astype(np.float64)
        K_np = K.detach().cpu().numpy().astype(np.float64)
        mask_np = mask.detach().cpu().numpy()  # (B, N) bool

        for b in range(B):
            if not valid_enough[b].item():
                continue

            valid = mask_np[b]  # (N,) bool
            p3 = pts3d_np[b, valid]  # (M, 3)
            p2 = pts2d_np[b, valid]  # (M, 2)
            K_b = K_np[b]  # (3, 3)
            valid_ct = int(valid.sum())  # Python int for clean division

            # Note: cv2.SOLVEPNP_SQPNP used since it works better when initial guess is not provided
            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=p3,
                    imagePoints=p2,
                    cameraMatrix=K_b,
                    distCoeffs=None,
                    iterationsCount=self.max_iters,
                    reprojectionError=self.inlier_thresh,
                    confidence=self.confidence,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
            except cv2.error:
                continue

            if not success or rvec is None:
                # cv2 failed entirely — leave fallback values in place.
                continue

            # Flatten inlier indices once; used for both refit and scoring.
            # cv2 may return inliers=None even on success (rare, means all
            # points were treated as inliers), so fall back to full range.
            inlier_idx = inliers.flatten() if inliers is not None else np.arange(valid_ct)
            n_inliers = len(inlier_idx)

            # Score is set immediately after RANSAC, before the refit,
            # so it always reflects the RANSAC inlier count regardless of
            # whether refinement runs or succeeds.
            score[b] = n_inliers / valid_ct

            # Optional LM refit on inliers for improved accuracy.
            # Wrapped in try/except because solvePnPRefineLM can raise on
            # degenerate inlier configurations (e.g. all-coplanar points).
            # On failure the RANSAC rvec/tvec are kept unchanged.
            if self.refine and n_inliers >= 4:
                try:
                    rvec, tvec = cv2.solvePnPRefineLM(
                        objectPoints=p3[inlier_idx],
                        imagePoints=p2[inlier_idx],
                        cameraMatrix=K_b,
                        distCoeffs=None,
                        rvec=rvec,
                        tvec=tvec,
                    )
                except cv2.error:
                    pass  # keep RANSAC result

            R_np, _ = cv2.Rodrigues(rvec)  # (3, 3)

            t_candidate = torch.from_numpy(tvec.flatten()).to(dtype)
            R_candidate = torch.from_numpy(R_np).to(dtype)

            # Guard against float64→float32 overflow and numerically diverged
            # solutions. solvePnPRansac/RefineLM can return success=True with
            # exploded tvec when correspondences are near-degenerate; the
            # overflow only becomes inf after the float32 cast.
            if t_candidate.isinf().any() or t_candidate.isnan().any() or R_candidate.isnan().any():
                score[b] = 0.0
                continue

            best_R[b] = R_candidate
            best_t[b] = t_candidate

        return best_R.to(device), best_t.to(device), score.to(device)


if __name__ == "__main__":
    from pathlib import Path

    import numpy as np
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    from configs import DATA_DIR

    from ..rendering.render import look_at_matrix
    from ..utils.camera import CameraIntrinsic

    torch.manual_seed(42)
    np.random.seed(42)

    T = look_at_matrix(np.array([[100, -30, 500]]), n_roll=3)[1]
    R = torch.from_numpy(T[:3, :3]).float()
    t = torch.from_numpy(T[:3, 3]).float()

    print(f"R: {R.round(decimals=2)}")
    print(f"t: {t.round(decimals=2)}")

    path = Path(DATA_DIR) / "lmo"
    mesh = trimesh.load(path / "models" / "obj_000008.ply")
    camera = CameraIntrinsic.from_json(path / "lmo" / "camera.json")

    # This is a hack to emulate the real data
    # We oversample tgt pts and filter for camera facing points
    pts3d = trimesh.sample.sample_surface(mesh, 1000)[0]
    pts3d_cam = pts3d @ R.numpy().T + t.numpy()

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

    mask = filter_camera_facing_points(mesh, T, pts3d_cam)
    pts3d = pts3d[mask]
    pts3d_cam = pts3d_cam[mask]

    print(f"pts3d: {pts3d.shape}", f"R: {R.shape}", f"t: {t.shape}")
    pts2d_h = pts3d_cam @ camera.K.T
    pts2d = pts2d_h[:, :2] / pts2d_h[:, 2:]

    pts3d = torch.tensor(pts3d, dtype=torch.float32).unsqueeze(0)
    pts2d = torch.tensor(pts2d, dtype=torch.float32).unsqueeze(0)
    K = torch.tensor(camera.K, dtype=torch.float32).unsqueeze(0)
    print(f"pts3d: {pts3d.shape}")
    print(f"pts2d: {pts2d.shape}")
    print(f"K: {K}")
    print(f"mask: {mask.sum()}")

    # Add noise to pts2d
    pts2d += torch.randn_like(pts2d) * 10.0
    # Add some outliers
    pts2d[0, :50] = (
        torch.randn(50, 2, device=pts2d.device, dtype=pts2d.dtype) * pts2d.std() + pts2d.mean()
    )

    def geodesic_error(R_est, R_gt):
        R_diff = torch.matmul(R_est.T, R_gt)
        trace = torch.trace(R_diff)
        cos_theta = (trace - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1, 1)
        return torch.acos(cos_theta)

    pose_estimator = PnPPoseEstimator(
        inlier_thresh=10.0, confidence=0.99, max_iters=1000, refine=True
    )
    print(pose_estimator)

    R_est, t_est, score = pose_estimator.estimate_pose(pts3d, pts2d, K)
    R_est, t_est, score = R_est[0], t_est[0], score[0]
    print(f"R_est: {R_est.round(decimals=2)}")
    print(f"t_est: {t_est.round(decimals=2)}")
    print(f"score: {score.item():.3f}")

    # Error between R_est and R
    geodesic_error_R = geodesic_error(R_est, R) / torch.pi * 180
    print(f"geodesic_error_R: {geodesic_error_R.item():.3f} degrees")

    # Error between t_est and t
    error_t = torch.norm(t_est - t) / torch.norm(t)
    print(f"error_t: {error_t.item():.3f}")
