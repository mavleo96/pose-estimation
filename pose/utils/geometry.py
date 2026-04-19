import numpy as np
import torch
from kornia.geometry.conversions import (
    euler_from_quaternion,
    quaternion_from_euler,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from trimesh.ray.ray_pyembree import RayMeshIntersector


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    roll, pitch, yaw: (B,)
    Note: convention is XYZ

    Returns:
        R: (B, 3, 3) rotation matrices
    """
    assert roll.ndim == pitch.ndim == yaw.ndim == 1
    assert roll.shape == pitch.shape == yaw.shape

    w, x, y, z = quaternion_from_euler(roll, pitch, yaw)  # XYZ convention
    q = torch.stack([w, x, y, z], dim=-1)  # (B, 4)
    R = quaternion_to_rotation_matrix(q)  # (B, 3, 3)
    return R


def rotation_matrix_to_euler(R):
    """
    R: (B, 3, 3) rotation matrices
    Note: convention is XYZ

    Returns:
        roll, pitch, yaw: each (B,)
    """
    assert R.ndim == 3 and R.shape[-2:] == (3, 3)

    q = rotation_matrix_to_quaternion(R)  # (B, 4) wxyz
    roll, pitch, yaw = euler_from_quaternion(q[..., 0], q[..., 1], q[..., 2], q[..., 3])
    return roll, pitch, yaw


def transform_from_Rt(R, t):
    """
    R: (B, 3, 3) rotation matrices
    t: (B, 3) translation vectors

    Returns:
        T: (B, 4, 4) transformation matrices, where T = [[R, t], [0, 0, 0, 1]]
    """
    assert R.ndim == 3 and R.shape[-2:] == (3, 3)
    assert t.ndim == 2 and t.shape[-1] == 3
    assert R.shape[0] == t.shape[0]

    T = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).expand(R.shape[0], -1, -1).clone()
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


def backproject_points(pts2d, depth, K):
    """
    find 3D points from 2D image points and depth

    pts2d: (..., N, 2)  2D image points in pixels (x, y)
    depth: (..., N)     depth values in mm
    K:     (..., 3, 3)  camera intrinsic matrix
    Returns:
           (..., N, 3)  3D points in camera space in mm
    """
    ones = torch.ones_like(pts2d[..., :1], device=pts2d.device, dtype=pts2d.dtype)
    pts2d_h = torch.cat([pts2d, ones], dim=-1)  # (..., N, 3)
    pts3d = torch.linalg.solve(K, pts2d_h.transpose(-2, -1))  # (..., 3, N)
    pts3d = pts3d.transpose(-2, -1)  # (..., N, 3)
    return pts3d * depth.unsqueeze(-1)


def apply_inverse_transform(pts3d, T):
    """
    find pts s.t. T * pts = pts3d

    pts3d: (..., N, 3)
    T:     (..., 4, 4)
    Returns:
           (..., N, 3)
    """
    ones = torch.ones_like(pts3d[..., :1])
    pts3d_h = torch.cat([pts3d, ones], dim=-1)  # (..., N, 4)
    result = torch.linalg.solve(T, pts3d_h.transpose(-2, -1))  # (..., 4, N)
    return result.transpose(-2, -1)[..., :3]  # (..., N, 3)


def get_visible_point_projections(points, model, K, T_m2c):
    """
    find pixel coordinates of 3D model points visible from a given camera pose

    casts a ray from the camera origin to each point and checks if the first mesh
    intersection occurs at the point itself — if not, the point is occluded.

    Args:
        points  : (N, 3) np.ndarray  — 3D model points in model space
        mesh    : trimesh.Trimesh    — mesh
        K       : (3, 3) np.ndarray  — camera intrinsics
        T_m2c   : (4, 4) np.ndarray  — model-to-camera transform

    Returns:
        point_mapping : (V,) np.int32   — indices of visible points into points
        pixel_coords  : (V, 2) np.int32 — (x, y) pixel coordinates of visible points
    """
    R = T_m2c[:3, :3]
    t = T_m2c[:3, 3]

    # transform points to camera coordinates
    p_cam = points @ R.T + t

    # direction of the points
    dirs = p_cam / np.linalg.norm(p_cam, axis=-1, keepdims=True)

    # get valid points for camera view by checking if intersection distance matches
    # distance from origin
    intersector = RayMeshIntersector(model.copy().apply_transform(T_m2c))
    locations, index_ray, _ = intersector.intersects_location(
        np.zeros_like(dirs), dirs, multiple_hits=False
    )
    visible_ray = np.full(points.shape[0], False)
    hit_dist = np.linalg.norm(locations, axis=1)
    point_dist = np.linalg.norm(p_cam, axis=1)
    visible_ray[index_ray] = np.abs(hit_dist - point_dist[index_ray]) < 1e-6

    # get pixel coordinates for valid points
    # point in camera coordinates @ adjusted camera matrix
    pixel_coords_homo = p_cam[visible_ray] @ K.T
    pixel_coords = (pixel_coords_homo[:, :2] / pixel_coords_homo[:, [2]]).astype(np.int64)
    point_mapping = np.argwhere(visible_ray).reshape(-1)

    return point_mapping, pixel_coords


__all__ = [
    "euler_to_rotation_matrix",
    "rotation_matrix_to_euler",
    "transform_from_Rt",
    "backproject_points",
    "apply_inverse_transform",
    "get_visible_point_projections",
]

if __name__ == "__main__":
    # euler to rotation matrix test
    print("euler_to_rotation_matrix test:")
    roll = torch.tensor([3 * torch.pi / 4, torch.pi / 2])
    pitch = torch.tensor([torch.pi / 6, torch.pi / 2])
    yaw = torch.tensor([torch.pi / 4, torch.pi / 2])
    print(
        f"roll[0]: {roll[0].item():.3f}, pitch[0]: {pitch[0].item():.3f},"
        f" yaw[0]: {yaw[0].item():.3f}"
    )
    print(
        f"roll[1]: {roll[1].item():.3f}, pitch[1]: {pitch[1].item():.3f},"
        f" yaw[1]: {yaw[1].item():.3f}"
    )

    R = euler_to_rotation_matrix(roll, pitch, yaw)

    print(f"R.shape: {R.shape}")
    print(f"R[0]: {torch.round(R[0], decimals=3)}")
    print(f"R[1]: {torch.round(R[1], decimals=3)}")

    # rot to rpy test
    print("\nrotation_matrix_to_euler test:")
    roll_est, pitch_est, yaw_est = rotation_matrix_to_euler(R)
    print(
        f"roll_est[0]: {roll_est[0].item():.3f}, pitch_est[0]: {pitch_est[0].item():.3f},"
        f" yaw_est[0]: {yaw_est[0].item():.3f}"
    )
    print(
        f"roll_est[1]: {roll_est[1].item():.3f}, pitch_est[1]: {pitch_est[1].item():.3f},"
        f" yaw_est[1]: {yaw_est[1].item():.3f}"
    )
