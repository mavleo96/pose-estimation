import numpy as np
import torch
from kornia.geometry.conversions import (
    euler_from_quaternion,
    quaternion_from_euler,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


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
    "euler_to_rotation_matrix",
    "rotation_matrix_to_euler",
    "transform_from_Rt",
    "get_crop_matrix",
    "get_pad_matrix",
    "get_resize_matrix",
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
