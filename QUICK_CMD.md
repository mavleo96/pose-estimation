python bop_toolkit/scripts/eval_bop19_pose.py \
    --renderer_type=vispy \
    --results_path /workspace/pose-estimation/results/large_debug/predictions \
    --eval_path /workspace/pose-estimation/results/large_debug \
    --result_filenames=pcr-pbrreal-rgb-mmodel_lmo-test_debug.csv \
    --num_workers=1


python scripts/error_analysis.py \
  --data_root=/workspace/pose-estimation/datasets \
  --dataset_name=lmo \
  --pred_csv=/workspace/pose-estimation/results/large_debug4/predictions/pcr-pbrreal-rgb-mmodel_lmo-test_debug4.csv


  python testing.py test_dataset_name=lmo run_id=large_debug_lmo +embedding_model=dinvo2 +dino_components=64 +embedding_desc=avg_normalized
