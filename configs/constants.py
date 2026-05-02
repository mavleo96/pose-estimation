# paths
DATA_DIR = "/data/pose-estimation"
RESULTS_DIR = "/workspace/pose-estimation/results"

# dataset names
BOP_CLASSIC = [
    "lmo",
    "tless",
    "itodd",
    "hb",
    "ycbv",
    "icbin",
    "tudl",
]
BOP_H3 = [
    "hot3d",
    "hope",
    "handal",
]
BOP_INDUSTRIAL = [
    "ipd",
    "xyzibd",
    "itoddmv",
]

# train datasets
TRAIN_DATASETS = ["gso", "shapenet"]

DETECTIONS_PATH_DICT = {
    "hb": "cnos-fastsam_hb-test_db836947-020a-45bd-8ec5-c95560b68011.json",
    "icbin": "cnos-fastsam_icbin-test_f21a9faf-7ef2-4325-885f-f4b6460f4432.json",
    "itodd": "cnos-fastsam_itodd-test_df32d45b-301c-4fc9-8769-797904dd9325.json",
    "lmo": "cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json",
    "tless": "cnos-fastsam_tless-test_8ca61cb0-4472-4f11-bce7-1362a12d396f.json",
    "tudl": "cnos-fastsam_tudl-test_c48a2a95-1b41-4a51-9920-a667cb3d7149.json",
    "ycbv": "cnos-fastsam_ycbv-test_f4f2127c-6f59-447c-95b3-28e1e591f1a1.json",
}
