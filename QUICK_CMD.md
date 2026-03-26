export RUN_ID="lmo_hard_dinov2"
python bop_toolkit/scripts/eval_bop19_pose.py \
    --renderer_type=vispy \
    --results_path /workspace/pose-estimation/results/large_${RUN_ID}/predictions \
    --eval_path /workspace/pose-estimation/results/large_${RUN_ID} \
    --result_filenames=pcr-pbrreal-rgb-mmodel_lmo-test_${RUN_ID}.csv \
    --num_workers=1

export RUN_ID="lmo_hard_dinov2"
python scripts/error_analysis.py \
  --data_root=/workspace/pose-estimation/datasets \
  --dataset_name=lmo \
  --pred_csv=/workspace/pose-estimation/results/large_${RUN_ID}/predictions/pcr-pbrreal-rgb-mmodel_lmo-test_${RUN_ID}.csv



python testing.py test_dataset_name=lmo run_id=lmo_hard_dinov2 +corres=hard +embedding_model=dinov2 +dino_components=64 +embedding_desc=avg_normalized
python testing.py test_dataset_name=lmo run_id=lmo_soft_dinov2 +corres=soft +embedding_model=dinov2 +dino_components=64 +embedding_desc=avg_normalized
python testing.py test_dataset_name=lmo run_id=lmo_hard_geo_dinov2 +corres=hard +embedding_model=dinov2 +dino_components=64 +embedding_desc=avg_normalized
python testing.py test_dataset_name=lmo run_id=lmo_soft_geo_dinov2 +corres=soft +embedding_model=dinov2 +dino_components=64 +embedding_desc=avg_normalized

python testing.py test_dataset_name=lmo run_id=lmo_soft_dinov2_32 +corres=soft +embedding_model=dinov2 +dino_components=32 +embedding_desc=avg_normalized
python testing.py test_dataset_name=lmo run_id=lmo_soft_dinov2_128 +corres=soft +embedding_model=dinov2 +dino_components=128 +embedding_desc=avg_normalized

python testing.py test_dataset_name=lmo run_id=lmo_hard_dinov3 +corres=hard +embedding_model=dinov3 +dino_components=64 +embedding_desc=avg_normalized
python testing.py test_dataset_name=lmo run_id=lmo_soft_dinov3 +corres=soft +embedding_model=dinov3 +dino_components=64 +embedding_desc=avg_normalized



## Download datasets
```bash
python scripts/download_bop_datasets.py --dataset_name=lmo
```

## Run PnP pipeline
```bash
python -m pose.inference \
    --run_name lmo_dinov2_pnp_518 \
    --dataset_name lmo \
    --detections_path /data/pose-estimation/bop_extra/default_detections/classic_bop23_model_based_unseen/cnos-fastsam/cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json \
    --device cuda \
    --dtype float16 \
    --output_size 518 518 \
    --batch_size 4 \
    --embedding_model dinov2
```