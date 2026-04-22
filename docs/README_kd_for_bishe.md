# Knowledge Distillation for smart-yun/bishe

## What this implementation does

This implementation adds teacher-student distillation on top of your existing
"prune first, finetune later" workflow:

- student: structurally pruned SegFormer-B1
- teacher: unpruned SegFormer-B1 checkpoint
- hard loss: original segmentation CE loss
- soft loss 1: logit KD on dense segmentation logits
- soft loss 2: CWD (channel-wise distillation) on one backbone feature map

The overall objective is:

L = L_seg + lambda_kd * L_logit_kd + lambda_cwd * L_cwd

## Why this is a good fit for this repo

Your repo already separates:
- `src/prune.py`
- `src/finetune.py`

So the cleanest place to add distillation is a new training entry:
- `src/finetune_kd.py`

This avoids rewriting pruning logic and keeps the experiment structure clean.

## Recommended setting for your project

For your current SegFormer-B1 experiments:

- start from **50% pruning** instead of 70%
- use **100 epoch** finetuning
- use **teacher = baseline B1 best checkpoint**
- start with **distill = logit+cwd**
- start with:
  - `kd_temperature = 4.0`
  - `kd_loss_weight = 1.0`
  - `cwd_tau = 1.0`
  - `cwd_loss_weight = 1.0`
  - `cwd_feature_index = -1`

## Important compatibility note

CWD is safest when teacher and student feature channels match.

That is true for your main pruning route:
- `mode = mlp_bottleneck`

Because that route mainly prunes internal FFN bottleneck channels and usually
keeps backbone stage output dimensions unchanged.

If you switch to:
- `mode = uniform_linear`
- aggressive global pruning

then CWD may fail because the feature channel dimensions can mismatch.
In that case:
- use `--distill logit`
- or add an explicit 1x1 alignment layer later.

## Files

- `src/distill_losses.py`
- `src/finetune_kd.py`
- `scripts/finetune_b1_p50_kd.sh`

## How to use

Copy the files into your repo:

- `src/distill_losses.py` -> `repo/src/distill_losses.py`
- `src/finetune_kd.py` -> `repo/src/finetune_kd.py`
- `scripts/finetune_b1_p50_kd.sh` -> `repo/scripts/finetune_b1_p50_kd.sh`

Then run:

```bash
bash scripts/finetune_b1_p50_kd.sh
```

## Expected behavior

A healthy 50% pruning + KD run should look like this:

- early stage: mIoU rises quickly from a badly damaged post-pruning model
- middle stage: KD usually gives a more stable recovery curve than plain finetune
- later stage: CWD often helps dense prediction more than plain logit KD alone

## Suggested ablations

Run these in order:

1. plain finetune
2. logit KD only
3. CWD only
4. logit KD + CWD

This gives you a clean thesis story:
- pruning hurts
- finetuning recovers
- KD further recovers dense prediction structure

## Caveat

This first implementation is intentionally easy to integrate.
For simplicity, it recomputes student features/logits once more inside the loss
function for distillation, so training is slower than a fully hook-optimized version.

If this version works well, the next step is to optimize it with feature hooks
so the student forward is not duplicated.
