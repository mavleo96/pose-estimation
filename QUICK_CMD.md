## Environment Setup
```bash
conda env create -f env.yml
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
python -m pose.inference.fusion_pipeline \
    --run_name lmo-fusion-kabsch-50-12-50 \
    --dataset_name lmo \
    --detections_path /data/pose-estimation/bop_extra/default_detections/classic_bop23_model_based_unseen/cnos-fastsam/cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json \
    --device cuda \
    --dtype bfloat16 \
    --batch_size 4 \
    --alpha 0.5 \
    --n_src_points 5000 \
    --n_tar_points 1000 \
    --dino_model_type dinov2 \
    --dino_output_size 224 224 \
    --dino_batch_size 20 \
    --dift_batch_size 4 \
    --up_block_indices 1 2 \
    --timestep 50 \
    --ensemble_size 4 \
    --sim_threshold 0.0 \
    --render_templates

python -m pose.inference.fusion2_pipeline \
    --run_name lmo-fusion2-kabsch-50-12-250 \
    --dataset_name lmo \
    --detections_path /data/pose-estimation/bop_extra/default_detections/classic_bop23_model_based_unseen/cnos-fastsam/cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json \
    --device cuda \
    --dtype bfloat16 \
    --batch_size 4 \
    --alpha 0.5 \
    --n_src_points 5000 \
    --n_tar_points 1000 \
    --dino_model_type dinov2 \
    --dino_output_size 224 224 \
    --dino_batch_size 20 \
    --dift_batch_size 4 \
    --up_block_indices 1 2 \
    --sim_threshold 0.0 \
    --render_templates

```

## BOP Evaluation

```bash
BOP_PATH=/data/pose-estimation python bop_toolkit/scripts/eval_bop19_pose.py \
    --renderer_type=vispy \
    --results_path /data/pose-estimation/results/lmo-fusion-kabsch-50-12-50 \
    --eval_path /data/pose-estimation/results/lmo-fusion-kabsch-50-12-50/bop_eval \
    --result_filenames=lmo-fusion-kabsch-50-12-50_lmo-test.csv \
    --num_workers=1 \
    --cleanup_eval
```
