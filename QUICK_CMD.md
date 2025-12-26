python bop_toolkit/scripts/eval_bop19_pose.py \
    --renderer_type=vispy \
    --results_path /workspace/pose-estimation/results/large_debug/predictions \
    --eval_path /workspace/pose-estimation/results/large_debug \
    --result_filenames=pcr-pbrreal-rgb-mmodel_lmo-test_debug.csv \
    --num_workers=1


python error_analysis.py \
  --scene_gt=/workspace/pose-estimation/datasets/lmo/test/000002/scene_gt.json \
  --pred_csv=/workspace/pose-estimation/results/large_lmo_pcr_dinvo2_normalized/predictions/pcr-pbrreal-rgb-mmodel_lmo-test_lmo_pcr_dinvo2_normalized.csv


python testing.py test_dataset_name=lmo run_id=large_debug_lmo +embedding_model=dinvo2 +dino_components=64 +embedding_desc=avg_normalized