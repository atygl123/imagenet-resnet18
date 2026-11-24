"""生成一个假数据集，用来在没有真数据时验证整条管线能跑通。

产物结构与真实 ImageNet 一致：
  <out>/train/<synset>/*.JPEG
  <out>/val/ILSVRC2012_val_00000001.JPEG ...（平铺）
  <out>/val_labels.txt              （每行：文件名 标签）

用法：python src/make_fake_data.py --out F:/tmp/fake
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

SYNSETS = ["n01440764", "n01443537", "n01484850", "n01491361",
           "n01494475", "n01496331", "n01498041", "n01514668"]


def noise_image(w, h, seed):
    rng = np.random.default_rng(seed)
    arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="F:/tmp/fake_imagenet")
    p.add_argument("--classes", type=int, default=8)
    p.add_argument("--train-per-class", type=int, default=24)
    p.add_argument("--val-per-class", type=int, default=4)
    a = p.parse_args()

    root = Path(a.out)
    classes = SYNSETS[:a.classes]
    seed = 0

    for ci, c in enumerate(classes):
        d = root / "train" / c
        d.mkdir(parents=True, exist_ok=True)
        for i in range(a.train_per_class):
            w, h = 180 + (i % 5) * 20, 200 + (i % 3) * 15   # 长宽比不同，测 resize
            noise_image(w, h, seed).save(d / f"{c}_{i:04d}.JPEG", quality=92)
            seed += 1

    val = root / "val"
    val.mkdir(parents=True, exist_ok=True)
    lines = []
    n = 0
    for ci, c in enumerate(classes):
        for j in range(a.val_per_class):
            n += 1
            name = f"ILSVRC2012_val_{n:08d}.JPEG"
            noise_image(220, 180, 90000 + n).save(val / name, quality=92)
            lines.append(f"{name} {ci}")

    (root / "val_labels.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成 {root}：{len(classes)} 类，训练 {len(classes)*a.train_per_class} 张，"
          f"验证 {n} 张")


if __name__ == "__main__":
    main()
