import numpy as np
import pyrender


# NOTE: this function has duplicate code with the generate_and_save_templates function in the
# dataset/render.py file
def render_view(
    mesh, T_m2c, K, width, height, depth_scale, ambient_light=0.2, directional_light=5.0
):
    """
    Render a view of a mesh from a given pose and camera intrinsics.

    Args:
        mesh          : trimesh.Trimesh — mesh
        T_m2c         : (4, 4)      — OpenCV world-to-camera matrix
        K             : (3, 3)      — camera intrinsics (fx, fy, cx, cy)
        width         : int         — image width
        height        : int         — image height
        depth_scale   : float       — divide depth by this to convert mm → dataset units
        ambient_light : float       — ambient light intensity
        directional_light : float   — directional light intensity

    Returns:
        rgb: (H, W, 3) RGB image
        depth: (H, W) depth image
    """
    # convert OpenCV m2c to OpenGL c2m
    # OpenCV: +X right, +Y down,  +Z forward
    # OpenGL: +X right, +Y up,    +Z backward
    # flip columns 1 and 2 of c2m to flip Y and Z camera axes
    pose_gl = np.linalg.inv(T_m2c) @ np.diag([1.0, -1.0, -1.0, 1.0])

    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 1], ambient_light=[ambient_light] * 3)
    scene.add(mesh)

    # znear/zfar set to cover 10mm to 10m
    cam_node = scene.add(
        pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=10.0, zfar=10000.0)
    )
    light_node = scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=directional_light))

    # update pose of camera and light
    scene.set_pose(cam_node, pose_gl)
    scene.set_pose(light_node, pose_gl)

    r = pyrender.OffscreenRenderer(width, height)
    rgb, depth = r.render(scene)
    r.delete()

    # scale depth: mm to dataset units, store as uint16 PNG
    # uint16 range 0–65535 covers 0–65535 units (65m at 1mm resolution)
    depth = (depth / depth_scale).astype(np.uint16)

    return rgb, depth
