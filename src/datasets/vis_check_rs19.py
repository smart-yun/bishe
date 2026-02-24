import random
from pathlib import Path
import numpy as np
from PIL import Image

SEED = 3407
N_SAMPLES = 50

def load_pairs(txt_path: Path):
    pairs = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        a, b = line.split()
        pairs.append((Path(a), Path(b)))
    return pairs

def make_palette(n=256, seed=SEED):
    rng = np.random.default_rng(seed)
    pal = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    pal[0] = np.array([0, 0, 0], dtype=np.uint8)
    return pal

def main():
    root = Path(__file__).resolve().parents[2]
    split = root / "data" / "splits" / "train.txt"
    out_dir = root / "results" / "vis_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(split)
    print(f"[INFO] loaded {len(pairs)} pairs from {split}")

    random.seed(SEED)
    random.shuffle(pairs)
    pairs = pairs[:N_SAMPLES]

    palette = make_palette(256, seed=SEED)

    bad_shape = 0
    bad_value = 0

    for i, (img_p, mask_p) in enumerate(pairs):
        img = Image.open(img_p).convert("RGB")
        mask = Image.open(mask_p)

        img_np = np.array(img)
        mask_np = np.array(mask)

        # mask 可能是 (H,W) 或 (H,W,3)，统一取单通道
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

        if img_np.shape[:2] != mask_np.shape[:2]:
            bad_shape += 1

        mn, mx = int(mask_np.min()), int(mask_np.max())
        if mn < 0 or mx >= 256:
            bad_value += 1

        mask_rgb = palette[mask_np.clip(0,255).astype(np.int64)]
        overlay = (img_np * 0.55 + mask_rgb * 0.45).astype(np.uint8)

        triplet = np.concatenate([img_np, mask_rgb, overlay], axis=1)
        Image.fromarray(triplet).save(out_dir / f"{i:03d}_{img_p.stem}.png")

    print(f"[INFO] bad_shape_count={bad_shape}, bad_value_count={bad_value}")
    print(f"[INFO] saved {N_SAMPLES} visuals to {out_dir}")

if __name__ == "__main__":
    main()
