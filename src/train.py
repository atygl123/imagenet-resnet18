"""训练入口。

设计原则：一次实验 = 一个 config + 一个 --out 目录，目录里必须能自证身份
（meta.json 含 commit / dirty / argv），否则三周后没人说得清哪组数字对应哪版代码。

示例：
  python src/train.py --config configs/e00_baseline.yaml --out runs/E00
  python src/train.py --resume runs/E00/last.pth --config configs/e00_baseline.yaml --out runs/E00
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from dataset import TrainImageFolder, ValImageFolder, build_train_transform, build_val_transform
from eval import evaluate
from utils import AverageMeter, JsonlLogger, set_seed, write_meta

DEFAULTS = {
    "data": "F:/data/imagenet",
    "cache": None,
    "val_labels": None,
    "epochs": 40,
    "batch_size": 256,
    "lr": 0.1,
    "momentum": 0.9,
    "wd": 1e-4,
    "res": 128,
    "label_smoothing": 0.0,
    "random_erasing": 0.0,
    "sched": "cosine",
    "warmup_epochs": 0,
    "seed": 42,
    "eval_every": 5,
    "workers": 6,
    "out": None,
    "resume": None,
    "max_iters": 0,          # 0 = 不限；冒烟时设 60
}


def load_config():
    """yaml 配置作为默认值，命令行参数可逐项覆盖。"""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args()
    cfg = dict(DEFAULTS)
    if known.config:
        with open(known.config, "r", encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})

    p = argparse.ArgumentParser("ImageNet 分类训练")
    for k, v in cfg.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k.replace('_', '-')}", type=lambda s: s.lower() == "true", default=v)
        elif v is None:
            p.add_argument(f"--{k.replace('_', '-')}", default=None)
        else:
            p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--config", default=known.config)
    a = p.parse_args()

    out = vars(a)
    out.pop("config", None)
    return out


def build_scheduler(opt, cfg):
    """warmup 用线性爬升，之后接 cosine 或 step。按 epoch 粒度。"""
    total, warm = cfg["epochs"], cfg["warmup_epochs"]

    def factor(epoch):
        if warm > 0 and epoch < warm:
            return (epoch + 1) / warm
        if cfg["sched"] == "cosine":
            t = (epoch - warm) / max(total - warm, 1)
            return 0.5 * (1 + math.cos(math.pi * t))
        progress = epoch / max(total, 1)
        if progress >= 8 / 9:
            return 0.01
        if progress >= 2 / 3:
            return 0.1
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def train_one_epoch(model, loader, crit, opt, scaler, device, epoch, log_every=50, max_iters=None):
    model.train()
    loss_m = AverageMeter()
    seen = 0
    t0 = time.perf_counter()
    for i, (x, y) in enumerate(loader):
        if max_iters and i >= max_iters:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            loss = crit(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        loss_m.update(loss.item(), x.size(0))
        seen += x.size(0)
        if log_every and (i + 1) % log_every == 0:
            print(f"    iter {i + 1}/{len(loader)}  loss {loss_m.avg:.4f}", end="\r")
    return loss_m.avg, time.perf_counter() - t0, seen


def main():
    cfg = load_config()
    if not cfg["out"]:
        raise SystemExit("必须指定 --out（例如 --out runs/E00）")
    os.makedirs(cfg["out"], exist_ok=True)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 数据 ----------
    train_root = cfg["cache"] or os.path.join(cfg["data"], "train")
    train_ds = TrainImageFolder(train_root, build_train_transform(cfg["res"], cfg["random_erasing"]))
    val_ds = ValImageFolder(os.path.join(cfg["data"], "val"), build_val_transform(cfg["res"]),
                            labels_file=cfg["val_labels"], class_names=train_ds.classes)

    pw = cfg["workers"] > 0
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["workers"], pin_memory=True,
                              persistent_workers=pw, prefetch_factor=2 if pw else None,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["workers"], pin_memory=True, persistent_workers=pw)

    # ---------- 模型 ----------
    # weights=None：任务书要求所有数字来自自己训练，不得使用预训练权重
    model = resnet18(weights=None, num_classes=len(train_ds.classes)).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                          weight_decay=cfg["wd"])
    sched = build_scheduler(opt, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    best_top1 = 0.0
    if cfg["resume"]:
        ck = torch.load(cfg["resume"], map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_top1 = ck.get("best_top1", 0.0)
        print(f"[resume] 从 epoch {start_epoch} 继续，best_top1={best_top1:.2f}")

    write_meta(cfg["out"], cfg, extra={
        "class_names": train_ds.classes,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "model": "resnet18",
        "pretrained": False,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "start_epoch": start_epoch,
    })
    logger = JsonlLogger(os.path.join(cfg["out"], "log.jsonl"))
    print(f"train {len(train_ds)} 张 | val {len(val_ds)} 张 | res {cfg['res']} | "
          f"bs {cfg['batch_size']} | epochs {cfg['epochs']} | seed {cfg['seed']}")

    for epoch in range(start_epoch, cfg["epochs"]):
        lr_now = opt.param_groups[0]["lr"]
        loss, sec, seen = train_one_epoch(model, train_loader, crit, opt, scaler, device, epoch,
                                          max_iters=cfg["max_iters"])
        sched.step()
        rec = {
            "epoch": epoch + 1,
            "train_loss": round(loss, 4),
            "lr": round(lr_now, 6),
            "epoch_sec": round(sec, 1),
            "img_per_s": round(seen / max(sec, 1e-6), 1),
            "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device.type == "cuda" else 0,
        }

        is_last = epoch + 1 == cfg["epochs"]
        if (epoch + 1) % cfg["eval_every"] == 0 or is_last:
            res = evaluate(model, val_loader, device, num_classes=len(train_ds.classes))
            rec.update({"val_top1": res["top1"], "val_top5": res["top5"]})
            if res["top1"] > best_top1:
                best_top1 = res["top1"]
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "cfg": cfg, "class_names": train_ds.classes,
                            "best_top1": best_top1},
                           os.path.join(cfg["out"], "best.pth"))
                rec["best"] = True

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch,
                    "cfg": cfg, "class_names": train_ds.classes,
                    "best_top1": best_top1},
                   os.path.join(cfg["out"], "last.pth"))
        logger.log(**rec)

    logger.close()
    with open(os.path.join(cfg["out"], "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"best_top1": best_top1, "cfg": cfg}, f, ensure_ascii=False, indent=2)
    print(f"\n完成。best_top1 = {best_top1:.2f}%，结果在 {cfg['out']}")


if __name__ == "__main__":
    main()
