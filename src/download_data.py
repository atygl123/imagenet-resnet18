"""从 HF 镜像并发分段下载 ImageNet parquet 分片（带断点续传）。

为什么不用 huggingface-cli / hf_transfer：
  - huggingface.co 本身不通，只能走 hf-mirror.com；
  - 镜像会 302 到 xethub CDN，单流约 200KB/s，分段并发能到 ~1MB/s；
  - 全量 23.4GB，实测约 6–7 小时，睡前挂上正好。

分片选择（重要）：
  - val   2 个 0.9GB  ← 必须先下，训练完要靠它算准确率
  - train 52 个 22.5GB
  - test  4 个 1.7GB   ← 默认跳过！这个数据集的 test 标签全是 -1，没有任何用处

用法：
  # 试探：只下 val（0.9GB，约 15 分钟），先打通管线
  python src/download_data.py --include val

  # 今晚正式挂机：val 优先，然后 train 全量
  python src/download_data.py --include val,train

断点续传：中断后重跑同一条命令即可，已完整的文件与已下完的分块都会跳过。
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MIRROR = "https://hf-mirror.com"
CHUNK = 16 * 1024 * 1024
PRIORITY = {"val": 0, "train": 1, "test": 2}


def api_tree(repo: str):
    """列 data/ 下的文件。镜像偶尔限流返回非列表，重试 3 次后报错退出。"""
    url = f"{MIRROR}/api/datasets/{repo}/tree/main/data"
    for attempt in range(1, 4):
        try:
            out = subprocess.check_output(["curl", "-s", "--max-time", "30", url], text=True)
            data = json.loads(out)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return [x for x in data if x.get("type") == "file"]
            print(f"  [第 {attempt} 次] 镜像返回异常：{str(data)[:80]}")
        except Exception as e:
            print(f"  [第 {attempt} 次] 解析失败：{e}")
        time.sleep(6 * attempt)
    raise SystemExit("镜像 API 持续异常（多半是限流）。请用 --files 直接指定文件名重试。")


def split_of(path: str) -> str:
    return Path(path).name.split("-")[0]


def file_url(repo: str, path: str) -> str:
    return f"{MIRROR}/datasets/{repo}/resolve/main/{path}"


def head_size(url: str) -> int:
    out = subprocess.check_output(
        ["curl", "-sIL", "--max-time", "30", url], text=True, stderr=subprocess.DEVNULL)
    size = 0
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            size = int(line.split(":")[1].strip())
    return size


def fetch_chunk(url, part, start, end, retry=3):
    for k in range(retry):
        r = subprocess.run(
            ["curl", "-sL", "--max-time", "600", "-r", f"{start}-{end}", "-o", part, url],
            capture_output=True)
        want = end - start + 1
        if r.returncode == 0 and Path(part).exists() and Path(part).stat().st_size == want:
            return True
        if Path(part).exists():
            Path(part).unlink()          # 截断的块直接丢掉重下
        time.sleep(2 + 3 * k)
    return False


def download_one(repo, item, out_dir: Path, threads: int):
    path = item["path"]
    dst = out_dir / Path(path).name
    total = item.get("size") or 0
    if dst.exists() and total and dst.stat().st_size == total:
        print(f"  [跳过] {dst.name} 已完整")
        return True

    url = file_url(repo, path)
    if not total:
        total = head_size(url)
    if not total:
        print(f"  [失败] 拿不到 {path} 的大小")
        return False

    print(f"  {dst.name}  {total / 1e9:.2f} GB  开始分段下载（{threads} 线程）")
    tmp = out_dir / (dst.name + ".parts")
    tmp.mkdir(exist_ok=True)
    ranges = [(i, min(i + CHUNK - 1, total - 1)) for i in range(0, total, CHUNK)]
    t0 = time.time()
    done = 0

    def job(rng):
        i, (a, b) = rng
        part = tmp / f"{i:08d}"
        if part.exists() and part.stat().st_size == b - a + 1:
            return part
        return part if fetch_chunk(url, part, a, b) else None

    with ThreadPoolExecutor(max_workers=threads) as ex:
        for res in ex.map(job, enumerate(ranges)):
            if res is None:
                print(f"  [失败] {dst.name} 有分块没下完，重跑本命令即可续传")
                return False
            done += 1

    with open(dst, "wb") as f:
        for i, _ in enumerate(ranges):
            f.write((tmp / f"{i:08d}").read_bytes())
    try:
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()
    except Exception as e:
        print(f"  [warn] 分块目录没清干净（不影响结果）：{e}")
    ok = dst.stat().st_size == total
    print(f"  {'完成' if ok else '大小不符'} {dst.name}  {dst.stat().st_size / 1e9:.2f} GB  "
          f"用时 {(time.time() - t0) / 60:.1f} 分钟")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="evanarlian/imagenet_1k_resized_256")
    p.add_argument("--out", default="F:/data/parquet")
    p.add_argument("--include", default="val,train", help="要下哪些划分，逗号分隔：val,train,test")
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--limit", type=int, default=None, help="只下前 N 个文件，试探用")
    p.add_argument("--files", default=None,
                   help="直接指定文件名（逗号分隔，在 data/ 下），可绕开被限流的镜像 API")
    a = p.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    want = {x.strip() for x in a.include.split(",") if x.strip()}

    if a.files:
        items = [{"path": f"data/{n.strip()}", "type": "file"} for n in a.files.split(",") if n.strip()]
        for it in items:
            it["size"] = head_size(file_url(a.repo, it["path"]))
    else:
        items = [x for x in api_tree(a.repo) if split_of(x["path"]) in want]
    items.sort(key=lambda x: (PRIORITY.get(split_of(x["path"]), 9), x["path"]))
    if a.limit:
        items = items[:a.limit]

    total = sum(x.get("size", 0) for x in items)
    free = shutil.disk_usage(out_dir).free / 1e9
    print(f"{a.repo}")
    print(f"  待下 {len(items)} 个文件，合计 {total / 1e9:.1f} GB → {out_dir}")
    print(f"  该盘剩余 {free:.0f} GB"
          f"{'  ← 空间不够！' if free * 1e9 < total * 1.2 else ''}\n")
    if a.include.find("test") >= 0:
        print("  提醒：test 划分的标签全是 -1，下它没有意义，除非你要复现官方测试集成绩\n")

    t_all = time.time()
    fails = []
    for i, it in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {it['path']}")
        if not download_one(a.repo, it, out_dir, a.threads):
            fails.append(it["path"])

    print(f"\n总计用时 {(time.time() - t_all) / 60:.1f} 分钟")
    if fails:
        print(f"有 {len(fails)} 个失败，重跑同一条命令可续传：{fails[:3]}")
        sys.exit(1)
    print("全部完成。下一步：python src/parquet_to_imagenet.py")


if __name__ == "__main__":
    main()
