from pathlib import Path
from PIL import Image

def main():
    root = Path(__file__).resolve().parents[2]
    in_dir = root / "results" / "vis_check_top10"
    out_dir = root / "results" / "vis_check_grid"
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(in_dir.glob("*.png"))
    assert len(imgs) >= 10, "Need at least 10 images in results/vis_check_top10"
    imgs = imgs[:10]

    thumb_w = 900
    thumbs = []
    for p in imgs:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        nh = int(h * thumb_w / w)
        thumbs.append(im.resize((thumb_w, nh)))

    cols, rows = 5, 2
    row_h = [max(thumbs[r*cols+c].size[1] for c in range(cols)) for r in range(rows)]
    W = cols * thumb_w
    H = sum(row_h)

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for r in range(rows):
        x = 0
        for c in range(cols):
            idx = r*cols + c
            canvas.paste(thumbs[idx], (x, y))
            x += thumb_w
        y += row_h[r]

    out_path = out_dir / "summary_grid.png"
    canvas.save(out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
