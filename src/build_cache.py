"""把训练图预缩成短边 146 的 JPEG 缓存。

为什么不直接存成 128×128 方图：
  那样训练时就没法做 RandomResizedCrop，增强多样性少一大半，基线会掉 2–3 个点。
  正确做法是保留长宽比、短边缩到 146，训练时再 RRC 到 128。
  146 = 128 / 0.875，与标准的 224→256 同一比例。

为什么值得做这一步：
  原始 JPEG 解码是每个 epoch 的固定开销。缓存后单 epoch 从分钟级降到秒级，
  7 组实验合计能省下好几个小时。

预计体积：128 万张 × 约 35KB ≈ 45GB
"""

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

SUFFIX = (".jpeg", ".jpg", ".png")


def _one(args):
    src, dst, short_side, quality = args
    try:
        img = Image.open(src).convert("RGB")
        w, h = img.size
        scale = short_side / min(w, h)
        if scale < 1.0:
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                             Image.BICUBIC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, "JPEG", quality=quality, optimize=False)
        return 1
    except Exception as e:
        print(f"[跳过] {src}: {e}")
        return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="F:/data/imagenet/train")
    p.add_argument("--dst", default="F:/cache/train_146")
    p.add_argument("--short-side", type=int, default=146)
    p.add_argument("--quality", type=int, default=90)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 张，调试用")
    a = p.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    jobs = []
    for cls_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in SUFFIX:
                jobs.append((img, dst / cls_dir.name / (img.stem + ".JPEG"),
                             a.short_side, a.quality))
    if a.limit:
        jobs = jobs[:a.limit]

    print(f"待处理 {len(jobs)} 张 → {dst}（短边 {a.short_side}，q={a.quality}）")
    t0 = time.perf_counter()
    done = 0
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, jobs, chunksize=64), 1):
            done += r
            if i % 20000 == 0:
                el = time.perf_counter() - t0
                print(f"  {i}/{len(jobs)}  {i / el:.0f} 张/秒  剩余约 {(len(jobs) - i) / (i / el) / 60:.1f} 分钟")

    el = time.perf_counter() - t0
    size_gb = sum(f.stat().st_size for f in dst.rglob("*.JPEG")) / 1024**3
    print(f"\n完成 {done} 张，用时 {el / 60:.1f} 分钟，{done / el:.0f} 张/秒，占用 {size_gb:.1f} GB")
    if os.environ.get("SHOW_DIR"):
        print(os.listdir(dst)[:5])


if __name__ == "__main__":
    main()
