

## PCR using kabsch
```bash
python testing.py test_dataset_name=lmo run_id=lmo_hard_dinov2 +corres=hard +embedding_model=dinov2 +dino_components=64 +embedding_desc=avg_normalized
```


## Download datasets
```bash
python scripts/download_bop_datasets.py --dataset_name=lmo
```

## Detection Paths

```bash
/data/pose-estimation/bop_extra/default_detections/classic_bop23_model_based_unseen/cnos-fastsam/

cnos-fastsam_hb-test_db836947-020a-45bd-8ec5-c95560b68011.json
cnos-fastsam_icbin-test_f21a9faf-7ef2-4325-885f-f4b6460f4432.json
cnos-fastsam_itodd-test_df32d45b-301c-4fc9-8769-797904dd9325.json
cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json
cnos-fastsam_tless-test_8ca61cb0-4472-4f11-bce7-1362a12d396f.json
cnos-fastsam_tudl-test_c48a2a95-1b41-4a51-9920-a667cb3d7149.json
cnos-fastsam_ycbv-test_f4f2127c-6f59-447c-95b3-28e1e591f1a1.json
```

## Run PnP pipeline
```bash
python -m pose.inference \
    --run_name lmo-dinov2-pnp-224 \
    --dataset_name lmo \
    --detections_path /data/pose-estimation/bop_extra/default_detections/classic_bop23_model_based_unseen/cnos-fastsam/cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json \
    --device cuda \
    --dtype float16 \
    --output_size 224 224 \
    --batch_size 4 \
    --embedding_model dinov2
```

## BOP Evaluation

```bash
BOP_PATH=/data/pose-estimation python bop_toolkit/scripts/eval_bop19_pose.py \
    --renderer_type=vispy \
    --results_path /data/pose-estimation/results/lmo-dinov2-pnp-224 \
    --eval_path /data/pose-estimation/results/lmo-dinov2-pnp-224/bop_eval \
    --result_filenames=lmo-dinov2-pnp-224_lmo-test.csv \
    --num_workers=1
```
