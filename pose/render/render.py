import numpy as np


def look_at_w2c(cam_pos, target=None, roll=0.0):
    """
    OpenCV world-to-camera matrix: +X right, +Y down, +Z forward.

    Args:
        cam_pos : (3,) or (N, 3) camera position(s) in world space
        target  : (3,) or (N, 3) look-at point(s), default origin
        roll    : float or (N,)  in-plane rotation, radians (+ = clockwise)

    Returns:
        (4, 4) or (N, 4, 4) float64 w2c matrix
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    squeeze = cam_pos.ndim == 1
    cam_pos = np.atleast_2d(cam_pos)
    N = len(cam_pos)

    if target is None:
        target = np.zeros((N, 3))
    else:
        target = np.broadcast_to(np.asarray(target, np.float64), (N, 3)).copy()

    roll = np.broadcast_to(np.asarray(roll, np.float64), (N,))

    # unit vector from camera to target → camera +Z axis
    fwd = target - cam_pos
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)

    # use world-Z when fwd ≈ ±world-Y (pole singularity)
    is_pole = np.abs(fwd[:, 1:2]) > 0.999
    world_up = np.where(is_pole, [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])

    # fwd × world_up → +X (right); right × fwd → +Y (down), reorthogonalized
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    up = np.cross(right, fwd)

    # rotate right/up around fwd by roll — 2D rotation in the image plane
    c, s = np.cos(roll)[:, None], np.sin(roll)[:, None]
    right, up = c * right + s * up, -s * right + c * up

    # rows of R are the camera axes; t = -R @ cam_pos moves world origin to camera frame
    R = np.stack([right, up, fwd], axis=1)
    t = -(R @ cam_pos[..., None]).squeeze(-1)

    M = np.eye(4, dtype=np.float64)[None].repeat(N, axis=0)
    M[:, :3, :3] = R
    M[:, :3, 3] = t

    return M[0] if squeeze else M
