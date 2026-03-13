import numpy as np
import pyrender
from PIL import Image
from pathlib import Path

AMBIENT_LIGHT = 1.0
DIRECTIONAL_LIGHT = 3.0


def look_at_matrix(cam_pos, roll=1, target=None):
    """
    Compute OpenCV world-to-camera matrix from camera position and target point.

    Coordinate convention (OpenCV):
        +X right, +Y down, +Z forward (into scene)

    No in-plane rotation by default (roll=1): world-Z is used as up reference,
    (BOP convention: Z-up world frame), falls back to world-Y near poles
    (when forward ≈ ±world-Z).

    Args:
        cam_pos : (N, 3) camera position(s) in world space
        roll    : int              number of in-plane rotations evenly spaced
                                   over [0, 2π). roll=1 → upright only.
        target  : (3,) look-at point, default origin

    Returns:
        (N * roll, 4, 4) float64 w2c matrices
    """
    assert cam_pos.ndim == 2 and cam_pos.shape[1] == 3
    assert 0 < roll <= 36

    N = len(cam_pos)

    if target is None:
        target = np.zeros((N, 3), dtype=np.float64)
    else:
        target = np.broadcast_to(target, (N, 3)).copy()

    # generate k roll angles evenly spaced over [0, 2π)
    roll_angles = np.linspace(0, 2 * np.pi, roll, endpoint=False)  # (k,)

    # repeat each position/target for each roll angle
    cam_pos = np.repeat(cam_pos, roll, axis=0)  # (N, 3) -> (N*k, 3)
    target = np.repeat(target, roll, axis=0)  # (N, 3) -> (N*k, 3)
    roll = np.tile(roll_angles, N)  # (k,) -> (N*k,)

    # unit vector from camera to target -> camera +Z axis
    fwd = target - cam_pos
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)

    # pole singularity: fwd ≈ ±world-Z, fall back to world-Y as up reference
    is_pole = np.abs(fwd[:, [2]]) > 0.999
    world_up = np.where(is_pole, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]).astype(np.float64)

    # fwd × world_up → +X (right); fwd × right → +Y (down), reorthogonalized
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    down = np.cross(fwd, right)

    # rotate right/down around fwd by roll — 2D rotation in the image plane
    c, s = np.cos(roll)[:, None], np.sin(roll)[:, None]
    right, down = c * right + s * down, -s * right + c * down

    # rows of R are the camera axes; t = -R @ cam_pos moves world origin to camera frame
    R = np.stack([right, down, fwd], axis=1)
    t = -(R @ cam_pos[..., None]).squeeze(-1)

    M = np.eye(4, dtype=np.float64)[None].repeat(len(cam_pos), axis=0)
    M[:, :3, :3] = R
    M[:, :3, 3] = t

    return M


def generate_and_save_templates(trimesh_model, T_m2c, K, width, height, depth_scale, save_dir):
    """
    Generate and save templates for a trimesh model from N poses and save RGB + depth to disk.

    Args:
        trimesh_model : trimesh.Trimesh — trimesh model
        T_m2c         : (N, 4, 4)   — OpenCV world-to-camera matrices
        K             : (3, 3)      — camera intrinsics (fx, fy, cx, cy)
        width         : int         — image width
        height        : int         — image height
        depth_scale   : float       — divide depth by this to convert mm → dataset units
        save_dir      : Path        — output directory

    Saves:
        save_dir/XXXX.png        (RGB, black background)
        save_dir/XXXX_depth.png  (uint16, scaled depth, 0=background)
    """
    assert save_dir is not None, "save_dir must be provided"
    assert isinstance(save_dir, Path), "save_dir must be a Path object"

    save_dir.mkdir(parents=True, exist_ok=True)

    mesh = pyrender.Mesh.from_trimesh(trimesh_model)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 1], ambient_light=[AMBIENT_LIGHT] * 3)
    scene.add(mesh)

    # znear/zfar set to cover 10mm to 10m
    cam_node = scene.add(
        pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=10.0, zfar=10000.0)
    )
    # light_node = scene.add(
    #     pyrender.DirectionalLight(color=[1, 1, 1], intensity=DIRECTIONAL_LIGHT)
    # )

    r = pyrender.OffscreenRenderer(width, height)
    for i, m2c in enumerate(T_m2c):
        # convert OpenCV m2c to OpenGL c2m
        # OpenCV: +X right, +Y down,  +Z forward
        # OpenGL: +X right, +Y up,    +Z backward
        # flip columns 1 and 2 of c2m to flip Y and Z camera axes
        pose_gl = np.linalg.inv(m2c) @ np.diag([1.0, -1.0, -1.0, 1.0])

        # update pose of camera and light
        scene.set_pose(cam_node, pose_gl)
        # scene.set_pose(light_node, pose_gl)

        rgb, depth = r.render(scene)

        # scale depth: mm to dataset units, store as uint16 PNG
        # uint16 range 0–65535 covers 0–65535 units (65m at 1mm resolution)
        depth_uint16 = (depth / depth_scale).astype(np.uint16)

        Image.fromarray(rgb).save(save_dir / f"{i:06d}.png")
        Image.fromarray(depth_uint16).save(save_dir / f"{i:06d}_depth.png")

    r.delete()
