python src/latency.py \
  --config-and-ckpt \
  configs/railsem19/segformer_b1_rs19_512x512_100ep_rtx4090.py \
  runs/B1_best_mIoU.pth \
  --device cuda:0 \
  --shape 512 512 \
  --batch-size 1 \
  --repeat 300 \
  --output-json output/segformer_b1_baseline/latency_bs1.json