python src/flops.py \
  --config-and-ckpt \
  configs/railsem19/segformer_b0_rs19_512x512_150ep_rtx4090.py \
  runs/B0_best_mIoU_v1.pth \
  --device cuda:0 \
  --shape 512 512 \
  --batch-size 1 \
  --output-json output/segformer_b0_baseline/flops_bs1.json