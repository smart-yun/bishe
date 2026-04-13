from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pruning/finetuning/eval/flops/latency experiments by experiment id."
    )
    parser.add_argument("--exp", required=True, help="Experiment id in configs/experiments.yaml")
    parser.add_argument(
        "--task",
        required=True,
        choices=["prune", "finetune", "eval", "flops", "latency"],
        help="Task to run",
    )
    parser.add_argument(
        "--variant",
        default=None,
        choices=["baseline", "pruned", "ft"],
        help="Used for eval/flops/latency. "
             "baseline = baseline checkpoint, "
             "pruned = pruned full model before finetune, "
             "ft = finetuned full model",
    )
    parser.add_argument(
        "--config-file",
        default="configs/experiments.yaml",
        help="Path to experiment yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print command, do not execute",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ratio_to_tag(ratio: float) -> str:
    return f"{int(round(float(ratio) * 100)):02d}"


def get_exp(cfg: dict[str, Any], exp_name: str) -> dict[str, Any]:
    experiments = cfg["experiments"]
    if exp_name not in experiments:
        raise KeyError(f"Experiment '{exp_name}' not found in yaml.")
    return experiments[exp_name]


def get_model_info(cfg: dict[str, Any], model_key: str) -> dict[str, Any]:
    models = cfg["globals"]["models"]
    if model_key not in models:
        raise KeyError(f"Model '{model_key}' not found in globals.models.")
    return models[model_key]


def resolve_best_checkpoint(ft_dir: Path) -> str:
    best_candidates = sorted(ft_dir.glob("best_mIoU*.pth"))
    if best_candidates:
        return str(best_candidates[-1])
    if (ft_dir / "latest.pth").exists():
        return str(ft_dir / "latest.pth")
    return str(ft_dir / "best_mIoU.pth")


def resolve_ft_full_model(ft_dir: Path) -> str:
    if (ft_dir / "best_full_model.pth").exists():
        return str(ft_dir / "best_full_model.pth")
    if (ft_dir / "latest_full_model.pth").exists():
        return str(ft_dir / "latest_full_model.pth")
    return str(ft_dir / "best_full_model.pth")


def get_paths(cfg: dict[str, Any], exp: dict[str, Any]) -> dict[str, str]:
    model_key = exp["model"]
    model_info = get_model_info(cfg, model_key)

    paths = {
        "config": model_info["config"],
        "checkpoint": model_info["checkpoint"],
        "baseline_output_dir": model_info["baseline_output_dir"],
    }

    if exp["type"] == "baseline":
        return paths

    ratio = float(exp["ratio"])
    tag = ratio_to_tag(ratio)

    suffix = ""
    if exp.get("global_pruning", False) and exp.get("isomorphic", False):
        suffix = "_gi"
    elif exp.get("global_pruning", False):
        suffix = "_g"

    run_tag = f"mlp{tag}{suffix}"

    paths["ratio"] = str(ratio)
    paths["tag"] = tag
    paths["run_tag"] = run_tag
    paths["pruned_output_dir"] = f"output/segformer_{model_key}_{run_tag}"
    paths["pruned_model"] = f"output/segformer_{model_key}_{run_tag}/model_pruned.pth"

    paths["ft_work_dir"] = f"runs/rs19/segformer_{model_key}_{run_tag}_ft"
    ft_dir = Path(paths["ft_work_dir"])

    paths["ft_checkpoint"] = resolve_best_checkpoint(ft_dir)
    paths["ft_full_model"] = resolve_ft_full_model(ft_dir)

    return paths


def default_variant(exp: dict[str, Any], task: str) -> str | None:
    if task in {"eval", "flops", "latency"}:
        if exp["type"] == "baseline":
            return "baseline"
        return "pruned"
    return None


def build_prune_cmd(cfg: dict[str, Any], exp: dict[str, Any], paths: dict[str, str]) -> list[str]:
    if exp["type"] == "baseline":
        raise ValueError("Prune task is not valid for baseline experiment.")

    gd = cfg["globals"]
    pd = gd["prune_defaults"]

    cmd = [
        sys.executable,
        "src/prune.py",
        "--config", paths["config"],
        "--checkpoint", paths["checkpoint"],
        "--device", gd["device"],
        "--shape", str(gd["shape"][0]), str(gd["shape"][1]),
        "--mode", pd["mode"],
        "--importance", pd["importance"],
        "--prune-stages", *[str(x) for x in pd["prune_stages"]],
        "--pruning-ratio", str(exp["ratio"]),
        "--iterative-steps", str(pd["iterative_steps"]),
        "--max-pruning-ratio", str(pd["max_pruning_ratio"]),
        "--round-to", str(pd["round_to"]),
        "--output-dir", paths["pruned_output_dir"],
    ]

    if exp.get("global_pruning", False):
        cmd.append("--global-pruning")

    if exp.get("isomorphic", False):
        cmd.append("--isomorphic")

    return cmd


def build_finetune_cmd(cfg: dict[str, Any], exp: dict[str, Any], paths: dict[str, str]) -> list[str]:
    if exp["type"] == "baseline":
        raise ValueError("Finetune task is not valid for baseline experiment.")

    fd = cfg["globals"]["finetune_defaults"]
    gd = cfg["globals"]

    epochs = exp.get("finetune_epochs", fd["epochs"])
    lr = exp.get("finetune_lr", fd["lr"])
    weight_decay = exp.get("finetune_weight_decay", exp.get("weight_decay", fd["weight_decay"]))

    cmd = [
        sys.executable,
        "src/finetune.py",
        "--config", paths["config"],
        "--pruned-model", paths["pruned_model"],
        "--device", gd["device"],
        "--work-dir", paths["ft_work_dir"],
        "--finetune-epochs", str(epochs),
        "--lr", str(lr),
        "--weight-decay", str(weight_decay),
    ]
    return cmd


def build_eval_cmd(cfg: dict[str, Any], exp: dict[str, Any], paths: dict[str, str], variant: str) -> list[str]:
    gd = cfg["globals"]

    if variant == "baseline":
        output_json = f"{paths['baseline_output_dir']}/metrics.json"
        work_dir = f"runs/tmp_eval_{exp['model']}_baseline"
        return [
            sys.executable,
            "src/eval.py",
            "--config-and-ckpt", paths["config"], paths["checkpoint"],
            "--device", gd["device"],
            "--work-dir", work_dir,
            "--output-json", output_json,
        ]

    if variant == "pruned":
        if exp["type"] == "baseline":
            raise ValueError("Pruned eval is not valid for baseline experiment.")
        output_json = f"{paths['pruned_output_dir']}/metrics_before_ft.json"
        work_dir = f"runs/tmp_eval_{exp['model']}_mlp{paths['tag']}"
        return [
            sys.executable,
            "src/eval.py",
            "--config", paths["config"],
            "--model", paths["pruned_model"],
            "--device", gd["device"],
            "--work-dir", work_dir,
            "--output-json", output_json,
        ]

    if variant == "ft":
        if exp["type"] == "baseline":
            raise ValueError("FT eval is not valid for baseline experiment.")
        output_json = f"{paths['pruned_output_dir']}/metrics_after_ft.json"
        work_dir = f"runs/tmp_eval_{exp['model']}_mlp{paths['tag']}_ft"
        return [
            sys.executable,
            "src/eval.py",
            "--config", paths["config"],
            "--model", paths["ft_full_model"],
            "--device", gd["device"],
            "--work-dir", work_dir,
            "--output-json", output_json,
        ]

    raise ValueError(f"Unknown eval variant: {variant}")


def build_flops_cmd(cfg: dict[str, Any], exp: dict[str, Any], paths: dict[str, str], variant: str) -> list[str]:
    gd = cfg["globals"]

    if variant == "baseline":
        output_json = f"{paths['baseline_output_dir']}/flops_bs1.json"
        return [
            sys.executable,
            "src/flops.py",
            "--config-and-ckpt", paths["config"], paths["checkpoint"],
            "--device", gd["device"],
            "--shape", str(gd["shape"][0]), str(gd["shape"][1]),
            "--batch-size", str(gd["batch_size"]),
            "--output-json", output_json,
        ]

    if variant == "pruned":
        if exp["type"] == "baseline":
            raise ValueError("Pruned flops is not valid for baseline experiment.")
        output_json = f"{paths['pruned_output_dir']}/flops_bs1.json"
        return [
            sys.executable,
            "src/flops.py",
            "--model", paths["pruned_model"],
            "--device", gd["device"],
            "--shape", str(gd["shape"][0]), str(gd["shape"][1]),
            "--batch-size", str(gd["batch_size"]),
            "--output-json", output_json,
        ]

    if variant == "ft":
        if exp["type"] == "baseline":
            raise ValueError("FT flops is not valid for baseline experiment.")
        output_json = f"{paths['pruned_output_dir']}/flops_after_ft_bs1.json"
        return [
            sys.executable,
            "src/flops.py",
            "--model", paths["ft_full_model"],
            "--device", gd["device"],
            "--shape", str(gd["shape"][0]), str(gd["shape"][1]),
            "--batch-size", str(gd["batch_size"]),
            "--output-json", output_json,
        ]

    raise ValueError(f"Unknown flops variant: {variant}")


def build_latency_cmd(cfg: dict[str, Any], exp: dict[str, Any], paths: dict[str, str], variant: str) -> list[str]:
    gd = cfg["globals"]

    if variant == "baseline":
        output_json = f"{paths['baseline_output_dir']}/latency_bs1.json"
        return [
            sys.executable,
            "src/latency.py",
            "--config-and-ckpt", paths["config"], paths["checkpoint"],
            "--device", gd["device"],
            "--shape", str(gd["shape"][0]), str(gd["shape"][1]),
            "--batch-size", str(gd["batch_size"]),
            "--repeat", str(gd["latency_repeat"]),
            "--output-json", output_json,
        ]

    if variant == "pruned":
        if exp["type"] == "baseline":
            raise ValueError("Pruned latency is not valid for baseline experiment.")
        output_json = f"{paths['pruned_output_dir']}/latency_bs1.json"
        return [
            sys.executable,
            "src/latency.py",
            "--model", paths["pruned_model"],
            "--device", gd["device"],
            "--shape", str(gd["shape"][0]), str(gd["shape"][1]),
            "--batch-size", str(gd["batch_size"]),
            "--repeat", str(gd["latency_repeat"]),
            "--output-json", output_json,
        ]

    if variant == "ft":
        if exp["type"] == "baseline":
            raise ValueError("FT latency is not valid for baseline experiment.")
        output_json = f"{paths['pruned_output_dir']}/latency_after_ft_bs1.json"
        return [
            sys.executable,
            "src/latency.py",
            "--model", paths["ft_full_model"],
            "--device", gd["device"],
            "--shape", str(gd["shape"][0]), str(gd["shape"][1]),
            "--batch-size", str(gd["batch_size"]),
            "--repeat", str(gd["latency_repeat"]),
            "--output-json", output_json,
        ]

    raise ValueError(f"Unknown latency variant: {variant}")


def build_cmd(cfg: dict[str, Any], exp_name: str, task: str, variant: str | None) -> tuple[list[str], str]:
    exp = get_exp(cfg, exp_name)
    paths = get_paths(cfg, exp)

    if variant is None:
        variant = default_variant(exp, task)

    if task == "prune":
        cmd = build_prune_cmd(cfg, exp, paths)
    elif task == "finetune":
        cmd = build_finetune_cmd(cfg, exp, paths)
    elif task == "eval":
        cmd = build_eval_cmd(cfg, exp, paths, variant)
    elif task == "flops":
        cmd = build_flops_cmd(cfg, exp, paths, variant)
    elif task == "latency":
        cmd = build_latency_cmd(cfg, exp, paths, variant)
    else:
        raise ValueError(f"Unknown task: {task}")


    return cmd, cfg["globals"]["project_root"]


def main():
    args = parse_args()
    cfg = load_yaml(args.config_file)

    cmd, cwd = build_cmd(cfg, args.exp, args.task, args.variant)

    print("Working directory:")
    print(cwd)
    print("Command:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, cwd=cwd, check=True)


if __name__ == "__main__":
    main()