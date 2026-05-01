from pathlib import Path
import argparse
import shutil


def collect_files(root, exts):
    root = Path(root)
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-root", default="data/railsem19/jpgs")
    parser.add_argument("--ann-root", default="data/railsem19/uint8")
    parser.add_argument("--out-root", default="deploy/val_subset")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    img_root = Path(args.img_root)
    ann_root = Path(args.ann_root)
    out_root = Path(args.out_root)

    out_img = out_root / "jpgs"
    out_ann = out_root / "uint8"

    if out_root.exists():
        shutil.rmtree(out_root)

    out_img.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)

    image_paths = collect_files(img_root, [".jpg", ".jpeg", ".png", ".bmp"])
    ann_paths = collect_files(ann_root, [".png", ".jpg", ".jpeg", ".bmp"])

    ann_by_stem = {}
    for p in ann_paths:
        ann_by_stem.setdefault(p.stem, p)

    pairs = []
    unmatched = []

    for img_path in image_paths:
        stem = img_path.stem

        candidate_stems = [
            stem,
            stem + "_label",
            stem + "_labelTrainIds",
            stem.replace("_leftImg8bit", "_gtFine_labelTrainIds"),
            stem.replace("_jpg", ""),
        ]

        ann_path = None
        for cand in candidate_stems:
            if cand in ann_by_stem:
                ann_path = ann_by_stem[cand]
                break

        if ann_path is None:
            unmatched.append(img_path)
            continue

        pairs.append((img_path, ann_path))

        if len(pairs) >= args.limit:
            break

    print(f"Total images: {len(image_paths)}")
    print(f"Total annotations: {len(ann_paths)}")
    print(f"Matched pairs: {len(pairs)}")
    print(f"Unmatched examples: {len(unmatched)}")

    if len(pairs) == 0:
        print("\nImage samples:")
        for p in image_paths[:10]:
            print("  IMG:", p)

        print("\nAnnotation samples:")
        for p in ann_paths[:10]:
            print("  ANN:", p)

        raise RuntimeError("No matched image-label pairs found.")

    for idx, (img_path, ann_path) in enumerate(pairs):
        new_stem = f"{idx:05d}_{img_path.stem}"

        img_suffix = img_path.suffix.lower()
        ann_suffix = ann_path.suffix.lower()

        dst_img = out_img / f"{new_stem}{img_suffix}"
        dst_ann = out_ann / f"{new_stem}{ann_suffix}"

        shutil.copy2(img_path, dst_img)
        shutil.copy2(ann_path, dst_ann)

        print(f"[{idx + 1:03d}] IMG: {img_path} -> {dst_img}")
        print(f"      ANN: {ann_path} -> {dst_ann}")

    print("\nDone.")
    print(f"Output images: {out_img}")
    print(f"Output labels: {out_ann}")


if __name__ == "__main__":
    main()
