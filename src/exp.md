# baseline 测量miou、flops、latency
python src/run_exp.py --exp b1_base --task eval # baseline evaluation miou
python src/run_exp.py --exp b1_base --task flops # baseline flops
python src/run_exp.py --exp b1_base --task latency # baseline latency

# 剪枝
python src/run_exp.py --exp b1_p10 --task prune # pruning with 10% ratio

# 微调前测miou、flops、latency
python src/run_exp.py --exp b1_p10 --task eval --variant pruned 
python src/run_exp.py --exp b1_p10 --task flops --variant pruned
python src/run_exp.py --exp b1_p10 --task latency --variant pruned

# 微调
python src/run_exp.py --exp b1_p10 --task finetune #finetuning after pruning

# 微调后测miou、flops、latency
python src/run_exp.py --exp b1_p10 --task eval --variant ft # evaluation of after finetune model
python src/run_exp.py --exp b1_p10 --task flops --variant ft
python src/run_exp.py --exp b1_p10 --task latency --variant ft # latency





# 全局剪枝
python src/run_exp.py --exp b1_p30_global --task prune

# 微调前测miou、flops、latency
python src/run_exp.py --exp b1_p30_global --task eval --variant pruned 
python src/run_exp.py --exp b1_p30_global --task flops --variant pruned
python src/run_exp.py --exp b1_p30_global --task latency --variant pruned

# 全局+iso
python src/run_exp.py --exp b1_p30_global_iso --task prune
