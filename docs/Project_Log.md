## **Project_Log**

---

## 02.23

今日完成：

bishe_push PAT




第一次提交流程

git config user.name "smart-yun"
git config user.email "你的邮箱@example.com"

git add -A
git commit -m "init: project scaffold"

git branch -M main
git push -u origin main

遇到的问题与解决



每次提交流程

git status  // 看看改了什么
git add -A //本次改动全部加入暂存区
git commit -m "docs: update weekly checklist" //形成一次版本记录
git push //推送github



明日计划



gpt上

重新安装torch 

数据集下载 跑起来



###################################################
编写数据索引生成脚本：输出 `train.txt`、`val.txt`、`test.txt`

0）先做 30 秒“目录与后缀确认”（避免索引配对失败）
cd ~/Projects/bishe

# 确认入口目录存在
ls -lah data/railsem19

# 统计图像/标签文件后缀（看 jpgs 是不是 .jpg，uint8 是不是 .png）
find data/railsem19/jpgs  -maxdepth 2 -type f | head -n 5
find data/railsem19/uint8 -maxdepth 2 -type f | head -n 5

如果看到：jpgs/*.jpg + uint8/*.png（最常见），继续下面步骤。
如果不是（比如标签是 .jpg），把后缀告诉我，你只需要改脚本里的 MASK_EXT 一行。

1）编写索引生成脚本（输出 train/val/test.txt）
1.1 写脚本：src/datasets/build_rs19_splits.py
cd ~/Projects/bishe
mkdir -p src/datasets data/splits

cat > src/datasets/build_rs19_splits.py <<'PY'
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
PY

1.2 运行索引脚本
conda activate railseg2
python src/datasets/build_rs19_splits.py

1.3 验收索引文件（产出物 1）
wc -l data/splits/train.txt data/splits/val.txt data/splits/test.txt
head -n 3 data/splits/train.txt

验收点：

三个文件都存在、行数 > 0

missing 不应接近 images（否则配对规则错）

#################################################################
2.1 写脚本：src/datasets/vis_check_rs19.py
cd ~/Projects/bishe
mkdir -p results/vis_check

cat > src/datasets/vis_check_rs19.py <<'PY'
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
PY

2.2 运行
conda activate railseg2
python src/datasets/vis_check_rs19.py
ls results/vis_check | wc -l


2.3验收标准：

bad_shape_count = 0

bad_value_count = 0（若不为 0，说明 mask 可能不是单通道 labelId，需要进一步解析 rs19-config.json，我也能继续带你做）

results/vis_check/ 下 50 张图


#################################################################
3 留痕：10 张代表性样例 + 1 页汇总拼图
3.1 选 10 张代表图（随机抽样即可）
mkdir -p results/vis_check_top10 results/vis_check_grid
rm -f results/vis_check_top10/*.png 2>/dev/null || true
ls results/vis_check | shuf -n 10 | xargs -I {} cp results/vis_check/{} results/vis_check_top10/
ls results/vis_check_top10 | wc -l

3.2 生成 1 页汇总拼图（2×5）
cat > src/datasets/make_grid_top10.py <<'PY'
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
PY

python src/datasets/make_grid_top10.py
ls -lh results/vis_check_grid/summary_grid.png

################################################################
4 最终验收与留痕（把输出放进 Experiment_Protocol.md）

你可以把下面命令的输出复制粘贴到 docs/Experiment_Protocol.md：

echo "===== RS19 SPLITS ====="
wc -l data/splits/train.txt data/splits/val.txt data/splits/test.txt
echo "===== RS19 VIS CHECK (last lines) ====="
python src/datasets/vis_check_rs19.py | tail -n 5
echo "===== TOP10 GRID ====="
ls -lh results/vis_check_grid/summary_grid.png

###################################################################
5、建议提交到 GitHub（只提交脚本 + splits + 汇总图，别提交大数据）
git add src/datasets/build_rs19_splits.py src/datasets/vis_check_rs19.py src/datasets/make_grid_top10.py data/splits results/vis_check_grid results/vis_check_top10
git commit -m "feat: add RailSem19 splits and GT overlay checks"
git push

不建议提交 results/vis_check/ 50 张（可留在本地）；若导师要求留痕，可只提交 top10 + summary_grid。