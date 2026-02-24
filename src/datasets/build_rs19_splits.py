import random
from pathlib import Path

SEED = 3407
IMG_EXT = ".jpg"
MASK_EXT = ".png"   # 如果 uint8 不是 png，在这里改成 .jpg 等

def main():
    root = Path(__file__).resolve().parents[2]  # bishe/
    data_root = root / "data" / "railsem19"
    img_dir = data_root / "jpgs"
    mask_dir = data_root / "uint8"

    assert img_dir.exists(), f"Missing: {img_dir}"
    assert mask_dir.exists(), f"Missing: {mask_dir}"

    imgs = sorted(img_dir.rglob(f"*{IMG_EXT}"))
    assert len(imgs) > 0, f"No {IMG_EXT} found under {img_dir}"

    pairs = []
    missing = 0
    for img in imgs:
        # 保持相对路径一致：jpgs/a/b.jpg -> uint8/a/b.png
        rel = img.relative_to(img_dir)
        mask = (mask_dir / rel).with_suffix(MASK_EXT)
        if not mask.exists():
            missing += 1
            continue
        pairs.append((img, mask))

    print(f"[INFO] images:  {len(imgs)}")
    print(f"[INFO] paired:  {len(pairs)}")
    print(f"[INFO] missing: {missing}")
    assert len(pairs) > 0, "No pairs found. Check MASK_EXT or naming."

    random.seed(SEED)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train = pairs[:n_train]
    val = pairs[n_train:n_train+n_val]
    test = pairs[n_train+n_val:]

    out_dir = root / "data" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(name, items):
        p = out_dir / f"{name}.txt"
        with p.open("w", encoding="utf-8") as f:
            for im, ma in items:
                f.write(f"{im.as_posix()} {ma.as_posix()}\n")
        print(f"[INFO] write {name}: {len(items)} -> {p}")

    dump("train", train)
    dump("val", val)
    dump("test", test)

if __name__ == "__main__":
    main()
