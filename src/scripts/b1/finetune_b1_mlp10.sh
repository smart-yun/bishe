python src/finetune.py \
  --config configs/railsem19/segformer_b1_rs19_512x512_100ep_rtx4090.py \
  --pruned-model output/segformer_b1_mlp10/model_pruned.pth \
  --device cuda:0 \
  --work-dir runs/rs19/segformer_b1_mlp10_ft_10epoch \
  --finetune-epochs 10 \
  --lr 3e-5

