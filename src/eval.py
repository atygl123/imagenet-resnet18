"""验证集评估。

两种用法：
  1. 命令行：python src/eval.py --ckpt runs/E00/last.pth --data F:/data/imagenet
  2. 被 train.py 调用：from eval import evaluate

除了 Top-1/Top-5，还会输出 1000 个类别的逐类准确率——
错误分析时靠它找"哪些类崩了"，不用再手动统计。
"""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import ValImageFolder, build_val_transform
from utils import accuracy


@torch.no_grad()
def evaluate(model, loader, device, num_classes=1000, want_preds=False, max_batches=None):
    model.eval()
    hit1 = hit5 = n = 0
    per_hit = [0] * num_classes
    per_cnt = [0] * num_classes
    preds = []
    t0 = time.perf_counter()

    for bi, (x, y, names) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)

        h1, h5 = accuracy(out, y, topk=(1, 5))
        hit1 += h1
        hit5 += h5
        n += y.numel()

        k = min(5, out.size(1))
        top5idx = out.topk(k, dim=1).indices.cpu()
        for i in range(y.numel()):
            t = int(y[i].item())
            per_cnt[t] += 1
            if int(top5idx[i, 0]) == t:
                per_hit[t] += 1
            if want_preds:
                preds.append({
                    "file": names[i],
                    "true": t,
                    "top5": [int(v) for v in top5idx[i]],
                    "correct": int(top5idx[i, 0]) == t,
                })

    elapsed = time.perf_counter() - t0
    per_class = [round(per_hit[c] / per_cnt[c], 4) if per_cnt[c] else None
                 for c in range(num_classes)]
    res = {
        "top1": round(100.0 * hit1 / max(n, 1), 4),
        "top5": round(100.0 * hit5 / max(n, 1), 4),
        "n_samples": n,
        "elapsed_sec": round(elapsed, 1),
        "img_per_s": round(n / max(elapsed, 1e-6), 1),
        "per_class_top1": per_class,
    }
    if want_preds:
        res["_preds"] = preds
    return res


def main():
    p = argparse.ArgumentParser("ImageNet 验证集评估")
    p.add_argument("--ckpt", required=True, help="checkpoint 路径")
    p.add_argument("--data", required=True, help="数据集根目录（含 val/）")
    p.add_argument("--val-labels", default=None, help="val 平铺结构时的标签文件")
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default=None, help="metrics.json 输出目录，默认与 ckpt 同级")
    p.add_argument("--save-preds", action="store_true", help="额外输出逐张预测，供错误分析用")
    p.add_argument("--max-batches", type=int, default=None, help="只跑前 N 个 batch，调试用")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    class_names = ckpt.get("class_names")

    from torchvision.models import resnet18
    model = resnet18(weights=None, num_classes=len(class_names) if class_names else 1000)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    ds = ValImageFolder(os.path.join(a.data, "val"), build_val_transform(a.res),
                        labels_file=a.val_labels, class_names=class_names)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                        pin_memory=True, persistent_workers=a.workers > 0)

    res = evaluate(model, loader, device, num_classes=len(ds.classes),
                   want_preds=a.save_preds, max_batches=a.max_batches)
    preds = res.pop("_preds", None)

    out_dir = a.out or os.path.dirname(os.path.abspath(a.ckpt))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics_eval.json"), "w", encoding="utf-8") as f:
        json.dump({**res, "ckpt": a.ckpt, "cfg": cfg}, f, ensure_ascii=False, indent=2)
    if preds:
        with open(os.path.join(out_dir, "preds.jsonl"), "w", encoding="utf-8") as f:
            for r in preds:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Top-1 {res['top1']:.2f}%  Top-5 {res['top5']:.2f}%  "
          f"n={res['n_samples']}  {res['img_per_s']} img/s")
    print(f"已写入 {os.path.join(out_dir, 'metrics_eval.json')}")


if __name__ == "__main__":
    main()
