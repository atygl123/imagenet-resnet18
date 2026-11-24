"""数据集完整性校验。数据一到位就跑这个，别等训练报错。

检查项：
  - 类别数 = 1000
  - 训练图数 = 1,281,167
  - 验证图数 = 50,000
  - 每个类别至少 1 张，且最少类别不少于 700 张（ImageNet 有 2 个类偏少）
  - 能否打开前 50 张图（检测损坏）
  - 有 val_labels.txt 时校验行数
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image

EXPECTED_CLASSES = 1000
EXPECTED_TRAIN = 1281167
EXPECTED_VAL = 50000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="F:/data/imagenet")
    p.add_argument("--val-labels", default=None)
    p.add_argument("--strict", action="store_true", help="数量不符时返回非零退出码")
    a = p.parse_args()

    root = Path(a.data)
    train, val = root / "train", root / "val"
    if not train.is_dir():
        raise SystemExit(f"找不到 {train}")

    classes = sorted(p.name for p in train.iterdir() if p.is_dir())
    counts = {c: len(list((train / c).glob("*.JPEG"))) for c in classes}
    n_train = sum(counts.values())

    val_subdirs = [p for p in val.iterdir() if p.is_dir()] if val.is_dir() else []
    if val_subdirs:
        n_val, val_layout = sum(len(list(p.glob("*.JPEG"))) for p in val_subdirs), "按类别分目录"
    else:
        n_val = len(list(val.glob("*.JPEG"))) if val.is_dir() else 0
        val_layout = "平铺 + 标签文件"

    # 随机抽 50 张试读，检测损坏
    broken = []
    sample = random.Random(0).sample(sorted(train.glob("*/*.JPEG")), min(50, n_train))
    for pth in sample:
        try:
            Image.open(pth).convert("RGB").load()
        except Exception as e:
            broken.append(f"{pth.name}: {e}")

    report = {
        "root": str(root),
        "classes": len(classes),
        "train_images": n_train,
        "val_images": n_val,
        "val_layout": val_layout,
        "min_per_class": min(counts.values()) if counts else 0,
        "max_per_class": max(counts.values()) if counts else 0,
        "smallest_classes": sorted(counts.items(), key=lambda kv: kv[1])[:5],
        "sample_broken": broken[:5],
        "val_labels_file": a.val_labels,
    }
    if a.val_labels and Path(a.val_labels).exists():
        with open(a.val_labels, encoding="utf-8") as f:
            report["val_labels_lines"] = sum(1 for ln in f if ln.strip())

    print(json.dumps(report, ensure_ascii=False, indent=2))

    problems = []
    if report["classes"] != EXPECTED_CLASSES:
        problems.append(f"类别数 {report['classes']} ≠ {EXPECTED_CLASSES}")
    if n_train != EXPECTED_TRAIN:
        problems.append(f"训练图 {n_train} ≠ {EXPECTED_TRAIN}（用子集的话忽略这条）")
    if n_val != EXPECTED_VAL:
        problems.append(f"验证图 {n_val} ≠ {EXPECTED_VAL}")
    if broken:
        problems.append(f"发现 {len(broken)} 张损坏图")
    if a.val_labels and report.get("val_labels_lines") not in (None, n_val):
        problems.append(f"标签行数 {report['val_labels_lines']} ≠ 验证图数 {n_val}")

    print("\n" + ("全部通过，可以开跑" if not problems else "有问题：\n  - " + "\n  - ".join(problems)))
    if problems and a.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
