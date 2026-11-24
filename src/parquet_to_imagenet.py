"""把 parquet 分片解包成标准 ImageNet 目录结构。

产物：
  <out>/train/<类别编号 5 位>/<原文件名>.JPEG
  <out>/val/<原文件名>.JPEG
  <out>/val_labels.txt        每行「文件名 标签」，按文件名排序

为什么类别目录用数字编号而不是 synset 名：
  parquet 里只有 int 标签，没有 synset。用 00000–00999 补零编号，
  排序后恰好等于标签顺序，和 train.py 的 ImageFolder 逻辑天然对齐。

提速技巧：parquet 里存的本来就是 JPEG 原始字节，直接写盘，不解码不重编码，
  23GB 大约 3–5 分钟；只有遇到 PNG 等格式才动用 PIL。

用法：
  python src/parquet_to_imagenet.py --split val
  python src/parquet_to_imagenet.py --split train
"""

import argparse
import io
import re
import time
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def synsets_from_classes_py(path: Path):
    """从 classes.py 里按出现顺序抽出 1000 个 synset。索引即标签。"""
    txt = path.read_text(encoding="utf-8")
    found = re.findall(r'"(n\d{8})"\s*:', txt)
    return found[:1000] if found else None


def is_jpeg(b: bytes) -> bool:
    return len(b) > 3 and b[0] == 0xFF and b[1] == 0xD8


def convert_shard(args):
    """处理单个分片，返回 (写入数, 跳过数, val标签列表)。可被多进程调用。"""
    f, split, out, limit = args
    pqfile = pq.ParquetFile(f)
    wrote = skipped = 0
    labels = []
    counter = 0
    for batch in pqfile.iter_batches(batch_size=2000):
        for r in batch.to_pylist():
            if limit and counter >= limit:
                break
            counter += 1
            img = r.get("image")
            if isinstance(img, dict):
                raw = img.get("bytes") or b""
                name = Path(img.get("path") or "").name
            else:
                raw, name = img or b"", ""
            label = r.get("label")

            if label is None or label < 0 or not raw:
                skipped += 1          # test 划分的 -1 标签
                continue
            if not name:
                name = f"{Path(f).stem}_{counter:06d}.JPEG"

            if split == "train":
                d = out / "train" / f"{label:05d}"
            else:
                d = out / "val"
            dst = d / name
            if dst.exists():
                skipped += 1
                continue
            d.mkdir(parents=True, exist_ok=True)
            if is_jpeg(raw):
                dst.write_bytes(raw)          # 原始字节直接落盘，不重编码
            else:
                Image.open(io.BytesIO(raw)).convert("RGB").save(dst, "JPEG", quality=95)
            wrote += 1
            if split != "train":
                labels.append((name, label))
        if limit and counter >= limit:
            break
    return wrote, skipped, labels


def convert(split: str, parquet_dir: Path, out: Path, limit: int = None, workers: int = 6,
            skip_last: int = 0):
    files = sorted(parquet_dir.glob(f"{split}-*.parquet"))
    if skip_last:
        # 最后一个分片可能还在下载，读它会拿到残缺数据
        files = files[:len(files) - skip_last]
        print(f"跳过最后 {skip_last} 个（可能仍在下载）")
    if not files:
        raise SystemExit(f"{parquet_dir} 里没有 {split}-*.parquet")

    bar = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    print(f"{split}: {len(files)} 个分片，共 {bar} 行，{workers} 进程")

    t0 = time.perf_counter()
    wrote = skipped = 0
    all_labels = []
    jobs = [(f, split, out, limit) for f in files]
    with Pool(workers) as pool:
        for i, (w, sk, labs) in enumerate(pool.imap_unordered(convert_shard, jobs), 1):
            wrote += w
            skipped += sk
            all_labels.extend(labs)
            el = time.perf_counter() - t0
            print(f"  [{i}/{len(files)}] +{w} 张  累计 {wrote} 张  "
                  f"{wrote / max(el, 1e-6):.0f} 张/秒  {Path(jobs[i - 1][0]).name}")

    if split == "val":
        lines = sorted(all_labels)
        (out / "val_labels.txt").write_text(
            "\n".join(f"{n} {lab}" for n, lab in lines) + "\n", encoding="utf-8")
        print(f"  已写 val_labels.txt（{len(lines)} 行）")

    el = time.perf_counter() - t0
    print(f"完成：写入 {wrote} 张，跳过 {skipped} 张，用时 {el / 60:.1f} 分钟")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", default="F:/data/parquet")
    p.add_argument("--out", default="F:/data/imagenet")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=None, help="每个分片只转前 N 行，测试用")
    p.add_argument("--workers", type=int, default=6, help="并行进程数")
    p.add_argument("--skip-last", type=int, default=0,
                   help="跳过最后 N 个分片（下载未结束时用，避免读到半个文件）")
    p.add_argument("--classes", default="assets/imagenet_classes.py",
                   help="类别定义文件，用来生成 classes.txt")
    a = p.parse_args()

    convert(a.split, Path(a.parquet_dir), Path(a.out), a.limit, a.workers, a.skip_last)

    # 顺手写一份「编号 → synset」对照，错误分析时用得上
    cls = Path(a.classes)
    if cls.exists():
        sy = synsets_from_classes_py(cls)
        if sy and len(sy) == 1000:
            (Path(a.out) / "classes.txt").write_text(
                "\n".join(f"{i:03d} {s}" for i, s in enumerate(sy)) + "\n", encoding="utf-8")
            print(f"已写 classes.txt（{len(sy)} 类，来自 {cls.name}）")
        else:
            print(f"[warn] {cls} 里只找到 {len(sy) if sy else 0} 个 synset，跳过 classes.txt")
    if a.split == "train":
        print("\n下一步：python src/check_data.py  （全部下完后才能通过校验）")


if __name__ == "__main__":
    main()
