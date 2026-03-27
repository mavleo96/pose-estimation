import json

import numpy as np
from bop_toolkit_lib.inout import save_bop_results

from .camera import CameraIntrinsic


def load_models_info(models_path):
    with open(models_path / "models_info.json") as f:
        models_info = json.load(f)
        models_info = {int(k): v for k, v in models_info.items()}
    return models_info


def load_camera_intrinsics(dataset_name, data_path):
    if dataset_name == "ycbv":
        # Note: YCBV dataset has two cameras: "uw" and "cmu"
        camera = CameraIntrinsic.from_json(data_path / "camera_uw.json")
    else:
        camera = CameraIntrinsic.from_json(data_path / "camera.json")
    return camera


def save_bop_results_from_predictions(method_name, dataset_name, split, save_dir, predictions_dir):
    """
    Save BOP results from predictions saved in the following format:

    The predictions should be available in the following format:
    {
        "scene_id": int,
        "im_id": int,
        "detection_time": float,
        "results": {
            "obj_id": list[int],
            "detection_score": list[float],
            "R": list[np.ndarray[float]],
            "t": list[np.ndarray[float]],
            "pose_estimation_time": list[float],
        }

    Args:
        method_name: Name of the method.
        dataset_name: Name of the dataset.
        split: Name of the split.
        save_dir: Path to the directory to save the results.
        predictions_dir: Path to the directory containing the predictions.
    """
    records = []
    for file in predictions_dir.glob("*.npz"):
        with np.load(file, allow_pickle=True) as data:
            info = {k: v.item() for k, v in data.items()}

        scene_id = info["scene_id"]
        im_id = info["im_id"]
        results = info["results"]

        time = info["detection_time"] + sum(results["pose_estimation_time"])

        for i in range(len(results["obj_id"])):
            records.append(
                {
                    "scene_id": scene_id,
                    "im_id": im_id,
                    "obj_id": results["obj_id"][i],
                    "score": results["detection_score"][i],
                    "R": results["R"][i],
                    "t": results["t"][i],
                    "time": time,
                }
            )
    save_path = save_dir / f"{method_name}_{dataset_name}-{split}.csv"
    save_bop_results(save_path, records)


__all__ = ["load_models_info", "load_camera_intrinsics", "save_bop_results_from_predictions"]
