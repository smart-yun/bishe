python src/latency.py \
  --model output/segformer_b1_mlp10/model_pruned.pth \
  --device cuda:0 \
  --shape 512 512 \
  --batch-size 1 \
  --repeat 300 \
  --output-json output/segformer_b1_mlp10/latency_bs1.json
