python src/flops.py \
  --model output/segformer_b1_mlp10/model_pruned.pth \
  --device cuda:0 \
  --shape 512 512 \
  --batch-size 1 \
  --output-json output/segformer_b1_mlp10/flops_bs1.json