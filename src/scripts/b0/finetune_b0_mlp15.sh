python src/finetune.py \
  --config configs/railsem19/segformer_b0_rs19_512x512_150ep_rtx4090.py \
  --pruned-model output/segformer_b0_mlp15/model_pruned.pth \
  --device cuda:0 \
  --work-dir runs/rs19/segformer_b0_mlp15_ft \
  --finetune-epochs 20 \
  --lr 3e-5 \
  --weight-decay 0.01