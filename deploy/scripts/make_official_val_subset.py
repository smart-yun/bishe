from pathlib import Path
import argparse
import shutil


def collect_files(root, exts):
    root = Path(root)
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def build_stem_map(files):
    out = {}
    for p in files:
        out.setdefault(p.stem, p)
    return out


def find_by_stem(stem, stem_map):
    candidates = [
        stem,
        Path(stem).stem,
        stem.replace("_leftImg8bit", ""),
        stem.replace("_gtFine_labelTrainIds", ""),
        stem.replace("_labelTrainIds", ""),
        stem.replace("_label", ""),
    ]

    # Also try opposite label-style expansions.
    extra = []
    for s in candidates:
        extra.extend([
            s,
            s + "_label",
            s + "_labelTrainIds",
            s + "_gtFine_labelTrainIds",
            s.replace("_leftImg8bit", "_gtFine_labelTrainIds"),
        ])

    for c in extra:
        if c in stem_map:
            return stem_map[c]

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", default="data/splits/val.txt")
    parser.add_argument("--img-root", default="data/railsem19/jpgs")
    parser.add_argument("--ann-root", default="data/railsem19/uint8")
    parser.add_argument("--out-root", default="deploy/val_official")
    args = parser.parse_args()

    split_file = Path(args.split_file)
    img_root = Path(args.img_root)
    ann_root = Path(args.ann_root)
    out_root = Path(args.out_root)

    out_img = out_root / "jpgs"
    out_ann = out_root / "uint8"

    if out_root.exists():
        shutil.rmtree(out_root)

    out_img.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)

    image_files = collect_files(img_root, [".jpg", ".jpeg", ".png", ".bmp"])
    ann_files = collect_files(ann_root, [".png", ".jpg", ".jpeg", ".bmp"])

    image_map = build_stem_map(image_files)
    ann_map = build_stem_map(ann_files)

    lines = [
        x.strip()
        for x in split_file.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

    pairs = []
    missing = []

    for line in lines:
        stem = Path(line).stem

        img_path = find_by_stem(stem, image_map)
        ann_path = find_by_stem(stem, ann_map)

        if img_path is None or ann_path is None:
            missing.append((line, img_path, ann_path))
            continue

        pairs.append((line, img_path, ann_path))

    print(f"Split entries: {len(lines)}")
    print(f"Available images: {len(image_files)}")
    print(f"Available labels: {len(ann_files)}")
    print(f"Matched pairs: {len(pairs)}")
    print(f"Missing pairs: {len(missing)}")

    if missing:
        print("\nMissing examples:")
        for item in missing[:20]:
            print(item)

    if not pairs:
        raise RuntimeError("No matched image-label pairs found.")

    for idx, (line, img_path, ann_path) in enumerate(pairs):
        original_stem = Path(line).stem
        new_stem = f"{idx:05d}_{original_stem}"

        dst_img = out_img / f"{new_stem}{img_path.suffix.lower()}"
        dst_ann = out_ann / f"{new_stem}{ann_path.suffix.lower()}"

        shutil.copy2(img_path, dst_img)
        shutil.copy2(ann_path, dst_ann)

    print("\nDone.")
    print(f"Output image dir: {out_img}")
    print(f"Output label dir: {out_ann}")


if __name__ == "__main__":
    main()
