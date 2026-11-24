"""数据读取。

两种用法：
  1. 直接读原始 train/val（不推荐，JPEG 解码是瓶颈）
  2. 读 build_cache.py 生成的短边 146 缓存，训练时再 RandomResizedCrop(128)

val 集兼容两种布局：
  - val/<synset>/*.JPEG        → 按目录分类（部分重打包版本）
  - val/ILSVRC2012_val_*.JPEG  → 需要 val_labels.txt
"""

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(res: int, random_erasing: float = 0.0):
    ops = [
        transforms.RandomResizedCrop(res, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if random_erasing > 0:
        # 放在 Normalize 之前：擦除的填充值应是 [0,1] 区间的像素值
        ops.append(transforms.RandomErasing(p=random_erasing, value="random"))
    ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


def build_val_transform(res: int):
    return transforms.Compose([
        transforms.Resize(int(res / 0.875)),   # 128 → 146，与训练缓存一致
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class TrainImageFolder(Dataset):
    """train/<类别>/*.JPEG 结构，类别按目录名排序编号。

    128 万个文件逐个 iterdir + sorted 要 5–6 分钟，而每次实验都要重建一次。
    所以第一次扫描后把清单写进 <root>/.index.tsv，之后直接读表，启动降到 1–2 秒。
    """

    def __init__(self, root, transform):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted(d.name for d in os.scandir(self.root) if d.is_dir())
        if not self.classes:
            raise FileNotFoundError(f"{root} 下没有类别子目录")
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        idx = self.root / ".index.tsv"
        if idx.exists():
            with open(idx, "r", encoding="utf-8") as f:
                self.samples = []
                for line in f:
                    rel, lab = line.rstrip("\n").rsplit("\t", 1)
                    self.samples.append((rel, int(lab)))   # 存字符串，别存 Path
            return

        self.samples = []
        for c in self.classes:
            cdir = self.root / c
            for e in os.scandir(cdir):
                if e.name.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((f"{c}/{e.name}", self.class_to_idx[c]))
        self.samples.sort()
        with open(idx, "w", encoding="utf-8") as f:
            for rel, lab in self.samples:
                f.write(f"{rel}\t{lab}\n")
        print(f"已建立文件索引 {idx}（{len(self.samples)} 条），下次启动会快很多")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rel, label = self.samples[i]
        with Image.open(os.path.join(self.root, rel)) as img:
            return self.transform(img.convert("RGB")), label


def _parse_label_file(path: str, class_to_idx: dict):
    """兼容四种常见的 val_labels.txt 格式。

    1. `n03444034 0`
    2. `0`                       （按文件名排序对应）
    3. `ILSVRC2012_val_00000001.JPEG n03444034`
    4. `ILSVRC2012_val_00000001.JPEG 0`
    """
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            toks = line.replace(",", " ").split()
            got = None
            # 末尾的整数优先
            if toks and toks[-1].lstrip("-").isdigit():
                v = int(toks[-1])
                got = v if 0 <= v < 1000 else None
            if got is None:
                # 找 synset 名
                for t in toks:
                    if t in class_to_idx:
                        got = class_to_idx[t]
                        break
            if got is None:
                raise ValueError(f"{path} 第 {lineno + 1} 行无法解析: {line!r}")
            labels.append(got)
    return labels


class ValImageFolder(Dataset):
    """支持两种 val 布局，返回 (image, label, filename)。"""

    def __init__(self, val_root, transform, labels_file=None, class_names=None):
        self.root = Path(val_root)
        self.transform = transform
        subdirs = [p for p in self.root.iterdir() if p.is_dir()]

        if subdirs:
            class_to_idx = {p.name: i for i, p in enumerate(sorted(p.name for p in subdirs))}
            self.samples = []
            for c in sorted(class_to_idx):
                for p in sorted((self.root / c).iterdir()):
                    if p.suffix.lower() in (".jpeg", ".jpg", ".png"):
                        self.samples.append((p, class_to_idx[c]))
            self.classes = sorted(class_to_idx)
        else:
            if not labels_file:
                raise ValueError("val 是平铺结构，必须提供 --val-labels 标签文件")
            files = sorted(p for p in self.root.iterdir()
                           if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
            if class_names is None:
                raise ValueError("平铺 val 需要传入 train 的类别名列表")
            class_to_idx = {c: i for i, c in enumerate(class_names)}
            labels = _parse_label_file(labels_file, class_to_idx)
            if len(labels) != len(files):
                raise ValueError(f"标签数 {len(labels)} 与图片数 {len(files)} 不一致")
            self.samples = list(zip(files, labels))
            self.classes = class_names

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, path.name
