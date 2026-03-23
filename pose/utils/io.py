import json

from .camera import CameraIntrinsic


def load_models_info(models_path):
    with open(models_path / "models_info.json") as f:
        models_info = json.load(f)
        models_info = {int(k): v for k, v in models_info.items()}
    return models_info


def load_camera_intrinsics(dataset_name, data_path):
    if dataset_name == "ycbv":
        # Note: YCBV dataset has two cameras: "uw" and "cmu"
        camera = CameraIntrinsic.from_json(data_path / dataset_name / "camera_uw.json")
    else:
        camera = CameraIntrinsic.from_json(data_path / dataset_name / "camera.json")
    return camera


__all__ = ["load_models_info", "load_camera_intrinsics"]
