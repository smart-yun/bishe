import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


try:
    from utils_mmseg import setup_mmseg

    setup_mmseg()
except Exception as exc:
    print(f"[WARN] setup_mmseg failed or not found: {exc}")


class SegFormerExportWrapper(torch.nn.Module):
    def __init__(self, model, height=512, width=512):
        super().__init__()
        self.model = model
        self.height = height
        self.width = width

    def forward(self, x):
        logits = self.model(x, data_samples=None, mode="tensor")

        if logits.shape[-2:] != (self.height, self.width):
            logits = F.interpolate(
                logits,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )

        return logits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    try:
        model = torch.load(args.model, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(args.model, map_location="cpu")

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected full torch.nn.Module, got {type(model)}")

    model.eval().to(device)

    wrapper = SegFormerExportWrapper(model, args.height, args.width)
    wrapper.eval().to(device)

    dummy = torch.randn(1, 3, args.height, args.width, device=device)

    with torch.no_grad():
        y = wrapper(dummy)
        print(f"[INFO] PyTorch output shape: {tuple(y.shape)}")

    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )

    print(f"[OK] Exported ONNX to: {output_path}")


if __name__ == "__main__":
    main()
