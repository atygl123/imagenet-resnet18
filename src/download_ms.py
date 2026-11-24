"""从魔搭（ModelScope）下载 ImageNet-1K 训练分片。

为什么训练走魔搭、验证走 HF：
  - 魔搭实测 7MB/s（单流即可，并发无增益），147GB 约 6 小时；
  - HF 镜像现在只有 0.3–0.9MB/s，23GB 要十几小时，赶不上明天的训练窗口；
  - 但魔搭这个仓库的 test 划分标签全是 -1，不能当验证集，
    所以验证集（0.9GB，50k 张带标签）从 HF 的 evanarlian 仓库补。

两者类别索引一致（都已核对：第 0 类都是 n01440764 tench），可以放心混用。

用法：
  python src/download_ms.py --split train --limit 3      # 先下 3 个分片试水
  python src/download_ms.py --split train                # 全部 294 个分片
  python src/download_ms.py --split test --limit 1       # 仅用于确认标签是 -1

断点续传：中断后重跑同一条命令，curl -C - 会接着下。
"""

import argparse
import subprocess
import time
from pathlib import Path

API = "https://www.modelscope.cn/api/v1/datasets/{repo}/repo?Revision=master&FilePath={path}"


def head_size(url: str) -> int:
    out = subprocess.check_output(
        ["curl", "-sIL", "--max-time", "30", url], text=True, stderr=subprocess.DEVNULL)
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":")[1].strip())
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="cutedataset/imagenet-1k")
    p.add_argument("--out", default="F:/data/parquet")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--total", type=int, default=None, help="该划分的分片总数，默认 train=294/test=28")
    p.add_argument("--limit", type=int, default=None, help="只下前 N 个，试水用")
    p.add_argument("--start", type=int, default=0)
    a = p.parse_args()

    total_shards = a.total or (294 if a.split == "train" else 28)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    idxs = list(range(a.start, total_shards))
    if a.limit:
        idxs = idxs[:a.limit]

    print(f"{a.repo}  {a.split}  共 {total_shards} 个分片，本次下 {len(idxs)} 个 → {out}")
    if a.split == "test":
        print("注意：该仓库 test 划分标签全是 -1，只能用来复现官方测试集，不能当验证集\n")

    t_all = time.time()
    got = 0
    print(f"开始时间 {time.strftime('%H:%M:%S')}，按 6.9MB/s 估算约 {len(idxs) * 0.5 / 6.9 / 3.6:.1f} 小时\n")
    for n, i in enumerate(idxs, 1):
        name = f"{a.split}-{i:05d}-of-{total_shards:05d}.parquet"
        dst = out / name
        url = API.format(repo=a.repo, path=f"data/{name}")

        size = head_size(url)
        if dst.exists() and size and dst.stat().st_size == size:
            print(f"[{n}/{len(idxs)}] {name} 已完整，跳过")
            got += 1
            continue
        if size == 0:
            print(f"[{n}/{len(idxs)}] {name} 拿不到大小，跳过")
            continue

        t0 = time.time()
        ok = False
        for attempt in range(1, 4):          # 6 小时挂机，必须能自己重试
            subprocess.run(["curl", "-sL", "-C", "-", "--max-time", "1800",
                            "-o", str(dst), url], capture_output=True)
            ok = dst.exists() and dst.stat().st_size == size
            if ok:
                break
            have = dst.stat().st_size / 1e9 if dst.exists() else 0
            print(f"     第 {attempt} 次未完成（已下 {have:.2f}/{size / 1e9:.2f} GB），续传重试…")
            time.sleep(5 * attempt)
        speed = dst.stat().st_size / max(time.time() - t0, 1e-6) / 1e6 if dst.exists() else 0
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [{n}/{len(idxs)}] {name}  {size / 1e9:.2f} GB  "
              f"{'完成' if ok else '仍失败，重跑可续传'}  {speed:.1f} MB/s")
        if ok:
            got += 1

    el = (time.time() - t_all) / 60
    print(f"\n完成 {got}/{len(idxs)} 个分片，用时 {el:.1f} 分钟")


if __name__ == "__main__":
    main()
