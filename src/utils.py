"""公共工具：随机种子、Git 信息、实验元数据、日志、指标。"""

import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int):
    """固定所有随机源。cudnn.deterministic 会略慢，但对照实验要求可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "N/A"


def git_info() -> dict:
    """commit hash + 工作区是否干净。dirty=True 的 run 不能进报告。"""
    return {
        "commit": _sh("git rev-parse --short HEAD"),
        "full_commit": _sh("git rev-parse HEAD"),
        "branch": _sh("git rev-parse --abbrev-ref HEAD"),
        "dirty": bool(_sh("git status --porcelain")),
    }


def env_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "argv": sys.argv,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["vram_gb"] = round(p.total_memory / 1024**3, 2)
    return info


def write_meta(out_dir: str, cfg: dict, extra: dict = None):
    """把这次实验的"身份"写进 out_dir/meta.json。"""
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **git_info(),
        **env_info(),
        "config": cfg,
    }
    if extra:
        meta.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if meta["dirty"]:
        print("[warn] 工作区有未提交改动，dirty=True —— 这个 run 的数字不能写进报告")


class JsonlLogger:
    """逐 epoch 追加一行 JSON。崩了也只丢最后一行，不会毁掉整份日志。"""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")

    def log(self, **kw):
        kw["ts"] = round(time.time(), 3)
        self._fh.write(json.dumps(kw, ensure_ascii=False) + "\n")
        self._fh.flush()
        print("  " + "  ".join(f"{k}={v}" for k, v in kw.items()))

    def close(self):
        self._fh.close()


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0


@torch.no_grad()
def accuracy(output, target, topk=(1, 5)):
    """返回 top-k 命中数列表（不是比例），由调用方累加。"""
    maxk = min(max(topk), output.size(1))
    _, pred = output.topk(maxk, dim=1)
    hit = pred.eq(target.view(-1, 1).expand_as(pred))
    return [hit[:, :min(k, maxk)].any(dim=1).sum().item() for k in topk]
