import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def load_gt(dataset_path):
    """
    Load ground-truth poses from dataset_path.
    Returns a mapping: scene_id -> {im_id -> {obj_id -> {R, t}}}
    """
    scene_info_paths = list(dataset_path.glob("test/*/scene_gt.json"))

    gt_poses = {}
    for scene_info_path in scene_info_paths:
        scene_id = int(scene_info_path.parent.name)

        with open(scene_info_path) as f:
            scene_gt = json.load(f)

        gt_poses[scene_id] = {
            int(im_id): {
                int(obj_info["obj_id"]): {
                    "R": np.array(obj_info["cam_R_m2c"]).reshape(3, 3),
                    "t": np.array(obj_info["cam_t_m2c"]),
                }
                for obj_info in im_info
            }
            for im_id, im_info in scene_gt.items()
        }

    return gt_poses


def load_predictions(predictions_path):
    """
    Load predictions from predictions_path.
    Returns a mapping: scene_id -> {im_id -> {obj_id -> {R, t}}}
    """
    pred_poses = {}
    with open(predictions_path) as f:
        reader = csv.DictReader(f)

        for row in reader:
            scene_id = int(row["scene_id"])
            im_id = int(row["im_id"])
            obj_id = int(row["obj_id"])

            R = parse_matrix(row["R"], 9).reshape(3, 3)
            t = parse_matrix(row["t"], 3)

            if scene_id not in pred_poses:
                pred_poses[scene_id] = {}
            if im_id not in pred_poses[scene_id]:
                pred_poses[scene_id][im_id] = {}
            pred_poses[scene_id][im_id][obj_id] = {"R": R, "t": t}

    return pred_poses


def parse_matrix(s, dim):
    vals = [float(x) for x in s.strip().split()]
    if len(vals) != dim:
        raise ValueError(f"Expected {dim} values, got {len(vals)} in '{s}'")
    return np.array(vals, dtype=float)


def compute_errors(gt_poses, pred_poses):
    errors = {}
    for scene_id, scene_preds in pred_poses.items():
        for im_id, im_preds in scene_preds.items():
            for obj_id, obj_pred in im_preds.items():
                if scene_id not in errors:
                    errors[scene_id] = {}
                if im_id not in errors[scene_id]:
                    errors[scene_id][im_id] = {}

                R_gt = gt_poses[scene_id][im_id][obj_id]["R"]
                t_gt = gt_poses[scene_id][im_id][obj_id]["t"]
                R_est = obj_pred["R"]
                t_est = obj_pred["t"]

                # TODO: Handle when gt is missing

                errors[scene_id][im_id][obj_id] = {
                    "R_err": geodesic_error(R_est, R_gt),
                    "t_err": translation_error(t_est, t_gt),
                }

    return errors


def geodesic_error(R_est, R_gt):
    R_rel = R_est @ R_gt.T
    trace = np.trace(R_rel)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, float(cos_theta)))
    return math.degrees(math.acos(cos_theta))


def translation_error(t_est, t_gt):
    return float(np.linalg.norm(t_est - t_gt))


def summarize_stats(errors):
    # Collect errors
    obj_error_summary = {}
    scene_error_summary = {}
    global_error_summary = {"R_err": [], "t_err": []}

    for scene_id, scene_errors in errors.items():
        for _im_id, im_errors in scene_errors.items():
            for obj_id, obj_errors in im_errors.items():
                if obj_id not in obj_error_summary:
                    obj_error_summary[obj_id] = {"R_err": [], "t_err": []}
                if scene_id not in scene_error_summary:
                    scene_error_summary[scene_id] = {"R_err": [], "t_err": []}
                obj_error_summary[obj_id]["R_err"].append(obj_errors["R_err"])
                obj_error_summary[obj_id]["t_err"].append(obj_errors["t_err"])
                scene_error_summary[scene_id]["R_err"].append(obj_errors["R_err"])
                scene_error_summary[scene_id]["t_err"].append(obj_errors["t_err"])
                global_error_summary["R_err"].append(obj_errors["R_err"])
                global_error_summary["t_err"].append(obj_errors["t_err"])

    # Summarise errors by object
    obj_error_summary = {
        obj_id: {
            "count": len(obj_errors["R_err"]),
            "R_err": (np.mean(obj_errors["R_err"]), np.std(obj_errors["R_err"])),
            "t_err": (np.mean(obj_errors["t_err"]), np.std(obj_errors["t_err"])),
        }
        for obj_id, obj_errors in obj_error_summary.items()
    }

    # Summarise errors by scene
    scene_error_summary = {
        scene_id: {
            "count": len(scene_errors["R_err"]),
            "R_err": (np.mean(scene_errors["R_err"]), np.std(scene_errors["R_err"])),
            "t_err": (np.mean(scene_errors["t_err"]), np.std(scene_errors["t_err"])),
        }
        for scene_id, scene_errors in scene_error_summary.items()
    }

    # Summarise errors globally
    global_error_summary = {
        "count": len(global_error_summary["R_err"]),
        "R_err": (
            np.mean(global_error_summary["R_err"]),
            np.std(global_error_summary["R_err"]),
        ),
        "t_err": (
            np.mean(global_error_summary["t_err"]),
            np.std(global_error_summary["t_err"]),
        ),
    }

    return obj_error_summary, scene_error_summary, global_error_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute rotation and translation errors per prediction"
    )
    parser.add_argument("--data_root", type=str, required=True, help="Path to data root")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--pred_csv",
        type=str,
        required=True,
        help="Path to prediction CSV in BOP format",
    )
    args = parser.parse_args()

    dataset_path = Path(args.data_root) / args.dataset_name

    gt_poses = load_gt(dataset_path)
    gt_count = sum([len(im) for scene in gt_poses.values() for im in scene.values()])
    print(f"Loaded {gt_count} ground-truth poses.")

    pred_poses = load_predictions(args.pred_csv)
    pred_count = sum([len(im) for scene in pred_poses.values() for im in scene.values()])
    print(f"Loaded {pred_count} predictions.\n")

    errors = compute_errors(gt_poses, pred_poses)
    obj_error_summary, scene_error_summary, global_error_summary = summarize_stats(errors)

    print("ERROR SUMMARY BY OBJECT")
    print(f"{'Object':<10} {'R_err (deg)':<20} {'t_err (mm)':<20} {'count':<10}")
    print("-" * 60)
    for obj_id, obj_errors in obj_error_summary.items():
        count = obj_errors["count"]
        r_mean, r_std = obj_errors["R_err"]
        t_mean, t_std = obj_errors["t_err"]

        r_str = f"{r_mean:.2f} ± {r_std:.2f}"
        t_str = f"{t_mean:.2f} ± {t_std:.2f}"

        print(f"{str(obj_id):<10} {r_str:<20} {t_str:<20} {count:<10}")
    print()

    print("ERROR SUMMARY BY SCENE")
    print(f"{'Scene':<10} {'R_err (deg)':<20} {'t_err (mm)':<20} {'count':<10}")
    print("-" * 60)
    for scene_id, scene_errors in scene_error_summary.items():
        count = scene_errors["count"]
        r_mean, r_std = scene_errors["R_err"]
        t_mean, t_std = scene_errors["t_err"]

        r_str = f"{r_mean:.2f} ± {r_std:.2f}"
        t_str = f"{t_mean:.2f} ± {t_std:.2f}"

        print(f"{str(scene_id):<10} {r_str:<20} {t_str:<20} {count:<10}")
    print()

    print("GLOBAL SUMMARY")
    print(f"{'R_err (deg)':<20} {'t_err (mm)':<20} {'count':<10}")
    print("-" * 60)

    count = global_error_summary["count"]
    r_mean, r_std = global_error_summary["R_err"]
    t_mean, t_std = global_error_summary["t_err"]

    r_str = f"{r_mean:.2f} ± {r_std:.2f}"
    t_str = f"{t_mean:.2f} ± {t_std:.2f}"

    print(f"{r_str:<20} {t_str:<20} {count:<10}")
    print()


if __name__ == "__main__":
    main()
