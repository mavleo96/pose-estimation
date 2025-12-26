import torch


def rpy_to_rot(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
) -> torch.Tensor:
    """
    roll, pitch, yaw: (B,)
    Returns:
        R: (B, 3, 3) rotation matrices
    """
    assert (
        roll.ndim == pitch.ndim == yaw.ndim == 1
    ), "roll, pitch, yaw must be 1D tensors"
    assert (
        roll.shape[0] == pitch.shape[0] == yaw.shape[0]
    ), "roll, pitch, yaw must have the same length"
    assert roll.shape[0] > 0, "roll, pitch, yaw must have at least one element"

    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    R = torch.zeros(roll.shape[0], 3, 3, device=roll.device, dtype=roll.dtype)

    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr

    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr

    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr

    return R


# TODO: pitch = 0 or pi/2 is not unique, need to handle this case
def rot_to_rpy(R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    R: (B, 3, 3) rotation matrices
    Returns:
        roll, pitch, yaw: each (B,)
    """
    assert R.ndim == 3 and R.shape[-2:] == (3, 3), "R must be a (B, 3, 3) tensor"
    assert R.shape[0] > 0, "R must have at least one element"

    pitch = torch.atan2(-R[:, 2, 0], torch.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2))

    roll = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    return roll, pitch, yaw


if __name__ == "__main__":
    # rpy to rot test
    print("rpy_to_rot test:")
    roll = torch.tensor([3 * torch.pi / 4, torch.pi / 2])
    pitch = torch.tensor([torch.pi / 6, torch.pi / 2])
    yaw = torch.tensor([torch.pi / 4, torch.pi / 2])
    print(
        f"roll[0]: {roll[0].item():.3f}, pitch[0]: {pitch[0].item():.3f}, yaw[0]: {yaw[0].item():.3f}"
    )
    print(
        f"roll[1]: {roll[1].item():.3f}, pitch[1]: {pitch[1].item():.3f}, yaw[1]: {yaw[1].item():.3f}"
    )

    R = rpy_to_rot(roll, pitch, yaw)

    print(f"R.shape: {R.shape}")
    print(f"R[0]: {torch.round(R[0], decimals=3)}")
    print(f"R[1]: {torch.round(R[1], decimals=3)}")

    # rot to rpy test
    print("rot_to_rpy test:")
    roll_est, pitch_est, yaw_est = rot_to_rpy(R)
    print(
        f"roll_est[0]: {roll_est[0].item():.3f}, pitch_est[0]: {pitch_est[0].item():.3f}, yaw_est[0]: {yaw_est[0].item():.3f}"
    )
    print(
        f"roll_est[1]: {roll_est[1].item():.3f}, pitch_est[1]: {pitch_est[1].item():.3f}, yaw_est[1]: {yaw_est[1].item():.3f}"
    )
