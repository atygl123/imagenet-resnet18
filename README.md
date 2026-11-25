# ImageNet 分类 · 训练与评估管线

ResNet-18 / ImageNet-1K / 128px / RTX 3060 Laptop 6GB。
配合 `../imagenet-schedule/` 里的排期表和三人执行手册使用。

## 一、环境

Python 环境在**隔壁目录**，不要重复装：

```bash
cd F:/WorkBuddy/workspace/2026-11-16-09-52-10/imagenet-proj
source ../imagenet-schedule/.venv/Scripts/activate    # Git Bash
# cmd 下用：..\imagenet-schedule\.venv\Scripts\activate.bat
```

已装：torch 2.14.1+cu126、torchvision 0.29.1、pillow、pyyaml、pyarrow。

## 二、目录

```
imagenet-proj/
├── src/
│   ├── download_ms.py     从魔搭下训练分片（快，7MB/s）
│   ├── download_data.py   从 HF 镜像下验证分片（慢，但只有 0.9GB）
│   ├── parquet_to_imagenet.py  解包成标准目录结构
│   ├── check_data.py      数据完整性校验（甲）
│   ├── build_cache.py     短边 146 缓存（丙）
│   ├── dataset.py         数据集与增强（乙）
│   ├── train.py           训练入口（乙）
│   ├── eval.py            评估与逐张预测（丙）
│   ├── utils.py           种子 / meta / 日志 / 指标
│   └── make_fake_data.py  无数据时验证管线
├── assets/imagenet_classes.py   1000 类的 synset 对照（来自数据集官方）
├── configs/               e00–e06 七组实验配置
└── runs/                  每次实验一个目录
```

## 三、数据从哪来（关键，先读这节）

官方 `image-net.org` 从本机不可达，HF 上的 `ILSVRC/imagenet-1k` 是 gated（403）。
实测过的三条可行路径：

| 来源 | 体积 | 实测速度 | 耗时 | 备注 |
|---|---|---|---|---|
| **魔搭** `cutedataset/imagenet-1k` | 147 GB | **6.9 MB/s** | ~6 h | 原始分辨率，test 划分标签全 -1 |
| HF 镜像 `evanarlian/imagenet_1k_resized_256` | 23.4 GB | 0.3–0.9 MB/s | 8–20 h | 已缩到 256px，train/val 都带标签 |
| 找人拷 | 166 GB | U 盘/移动硬盘 | 数小时 | 最快，且通常最可靠 |

**推荐组合：训练走魔搭，验证走 HF。**

理由：魔搭快一个数量级，但它的 test 划分标签全是 `-1`（无法当验证集）；
HF 那个仓库的 val 划分只有 0.9GB、标签完整，正好补上。
两者类别索引已核对一致（第 0 类都是 `n01440764 tench`），可以混用。

### 执行顺序

```bash
# 1) 验证集（0.9GB，约 30–50 分钟）
python src/download_data.py --include val --out F:/data/parquet
python src/parquet_to_imagenet.py --split val

# 2) 训练集（147GB，约 6 小时，睡前挂上）
python src/download_ms.py --split train --out F:/data/parquet
python src/parquet_to_imagenet.py --split train
```

两条下载命令都支持断点续传：中断后重跑同一条命令即可。

**磁盘账：** 147GB 分片 + 166GB 解包图 = 313GB。F 盘现有 438GB，够但别再多留一份。
紧张的话，转完一批就删同名的 parquet。

### 不想下 147GB 的替代方案

只下 HF 那个 256px 版本（23.4GB）：

```bash
python src/download_data.py --include val,train
python src/parquet_to_imagenet.py --split val
python src/parquet_to_imagenet.py --split train
```

好处是**不用再建 146 缓存**（图已经是 256px），磁盘只占 23GB。
代价是下载慢：网络好 8 小时，网络差 20 小时。

## 四、校验与缓存

```bash
# 校验：要看到 1000 类 / 1281167 训练图 / 50000 验证图
python src/check_data.py --data F:/data/imagenet --val-labels F:/data/imagenet/val_labels.txt

# 只有用魔搭原始分辨率时才需要建缓存（约 30 分钟，45GB）
python src/build_cache.py --src F:/data/imagenet/train --dst F:/cache/train_146 --workers 8
```

用 HF 的 256px 版本时跳过建缓存，把 configs 里的 `cache` 改成 `null`。

## 五、冒烟训练

**注意：不要跑满一个 epoch**——128 万张要十几分钟，而且第 1 个 iteration 就能看出问题。
用 `--max-iters` 只跑 60 步，30 秒出结果：

```bash
python src/train.py --config configs/e00_baseline.yaml \
  --epochs 1 --batch-size 64 --max-iters 60 --eval-every 1 --out runs/SMOKE
```

看三件事：loss 是否明显下降（前 60 步应从 ~7 降到 5 以下）、
显存是否 < 4GB、`meta.json` 里 `dirty` 是不是 false。

**另做一次 5 分钟的过拟合测试**（最容易发现标签映射 bug）：

```bash
python src/train.py --config configs/e00_baseline.yaml --epochs 60 --batch-size 16 \
  --random-erasing 0 --out runs/OVERFIT
```

做法：临时把训练集换成一个只有 16 张图的小目录。能记住到 100% 才算管线没问题；
做不到说明标签或损失有 bug，别往下跑。

## 六、七组正式实验

```bash
python src/train.py --config configs/e00_baseline.yaml  --out runs/E00   # 基线  甲
python src/train.py --config configs/e01_erasing.yaml   --out runs/E01   # 甲
python src/train.py --config configs/e02_ls.yaml        --out runs/E02   # 乙
python src/train.py --config configs/e03_sched.yaml     --out runs/E03   # 丙
python src/train.py --config configs/e04_combo.yaml     --out runs/E04   # 组合 小组
python src/train.py --config configs/e05_ablate.yaml    --out runs/E05   # 消融
python src/train.py --config configs/e06_seed123.yaml   --out runs/E06   # 复现 乙
```

每组约 **6.6 小时**（24 epoch × 15.8 分钟）。产物：`meta.json`（commit/dirty）、`log.jsonl`（逐 epoch）、`best.pth`、`summary.json`。

### 为什么是 24 epoch

实测端到端吞吐 **1309 img/s**（12 epoch 完整跑完测得；纯 GPU 上限 3048，瓶颈是每 epoch 解码 128 万个 JPEG）。
一个 epoch ≈ 15.8 分钟，因此：

| epoch | 单组 | 七组 | 一晚一组 |
|---|---|---|---|
| 12 | 3.3 h | 23 h | 4 晚跑完，但 val 仍在陡升 |
| **24** | **6.6 h** | **46 h** | **7 晚，余量 1.4 h/晚** |
| 40 | 11 h | 77 h | 跑不完 |

12 epoch 的实测轨迹是 36.5 → 52.7 → 55.1，**末轮仍在急升**，属于欠训练而不是收敛。
所以取 24 epoch：既能让曲线进入平台，又能在 8 小时夜间窗口内留出 1.4 小时的重跑余量。

代价要写进报告：基线 Top-1 预期 63–65%（论文 69.8%），差距来自"24 epoch + 128px"，
不是实现错误。这句话必须在报告第 2 节讲清楚。

中断续跑：

```bash
python src/train.py --config configs/e00_baseline.yaml --out runs/E00 --resume runs/E00/last.pth
```

## 七、评估与错误分析

```bash
python src/eval.py --ckpt runs/E04/best.pth --data F:/data/imagenet \
  --val-labels F:/data/imagenet/val_labels.txt --save-preds
```

产出 `metrics_eval.json`（Top-1/5 + 1000 类逐类准确率）和 `preds.jsonl`（逐张 top-5）。
把 `per_class_top1` 和 `classes.txt` 对照，就能定位最差的类，直接从那里挑错误案例。

## 八、一条硬规则

`meta.json` 里 **`dirty: true` 的 run，数字不能写进报告**。

```bash
git status                     # 必须干净
git add -A && git commit -m "run(E00): baseline"
python src/train.py --config configs/e00_baseline.yaml --out runs/E00
git rev-parse --short HEAD     # 抄进实验清单 L 列
```

## 九、代码与数据来源说明（任务书 §10.1）

### 9.1 网络结构

| 项 | 内容 |
|---|---|
| 来源 | `torchvision.models.resnet18`（torchvision 0.29.1） |
| 许可证 | **BSD-3-Clause**（torchvision 仓库，Copyright © PyTorch contributors） |
| 原始出处 | He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 |
| 我们的修改 | 仅将 `weights` 设为 `None`（随机初始化，不使用预训练权重），输出类别数设为 1000 |

### 9.2 训练 / 评估 / 数据管线

| 项 | 内容 |
|---|---|
| 来源 | 本项目自行实现（`src/train.py`、`src/eval.py`、`src/dataset.py`） |
| 流程参考 | PyTorch 官方 `references/classification/train.py`（BSD-3-Clause），仅借鉴流程组织 |
| 我们的修改 | 全部代码为本组编写；未使用 timm、mmclassification 等第三方训练框架 |

**注：** 未直接复制官方 `train.py`，因此不构成对原文件的修改，仅流程结构参考。

### 9.3 数据

| 项 | 内容 |
|---|---|
| 数据集 | ImageNet-1K（ILSVRC 2012），1000 类 |
| 使用规模 | 训练 1,281,167 张（全量）／验证 50,000 张（完整 val） |
| 实际下载渠道 | 魔搭（ModelScope）镜像 `cutedataset/imagenet-1k`；验证集来自 HF 镜像 `evanarlian/imagenet_1k_resized_256` 的 val 划分 |
| 许可证 | ImageNet 非商业研究与教育用途条款（Princeton/Stanford） |
| 预处理 | 训练图短边缩放至 146px 缓存；训练时 `RandomResizedCrop(128)`；验证 `Resize(146) + CenterCrop(128)` |
| 类别对照 | `assets/imagenet_classes.py` 中 1000 个 synset 与标签序号的映射 |

**必须在报告中注明：** 数据经第三方镜像获取，非 ImageNet 官方站点直接下载。
本项目的数字均为自训产出，未引用任何论文或他人报告的指标作为对照（符合 §10.3）。

## 十、已知限制

- 只支持 ResNet-18，换模型要改 `train.py` / `eval.py` 里建模型的两行。
- 单卡，无 DDP。
- 官方 test 集（10 万张）标签不公开或为 -1，本项目只用 val 的 5 万张。

## 十一、提交方式（Git 仓库链接）

### 仓库里必须有什么

| 要提交 | 不要提交 |
|---|---|
| `src/`、`configs/`、`assets/`、`README.md` | `data/`（166GB 原始图） |
| `runs/E0X/log.jsonl`、`meta.json`、`summary.json` | `cache/`（13.5GB 训练缓存） |
| `runs/E0X/metrics_eval.json`、`preds.jsonl` | `*.pth`（每个 45MB，评审用不上） |
| `report/` 里的分析文档与 PPT | `.venv/` |

**`.gitignore` 里千万不要整目录忽略 `runs/`。** 任务书要求"他人可以按照记录复现关键结果"，
如果实验日志一个都没进仓库，评审第 4 条（团队分工与可复现性）会直接失分。
现在的规则是只忽略 `*.pth`，日志与指标全部保留——单组实验的文本产物只有几十 KB。

### 推送步骤（GitHub）

1. 打开 [github.com/new](https://github.com/new)，仓库名如 `imagenet-resnet18`，
   选 **Public**，**不要**勾 "Add a README file / .gitignore / license"（保持空仓库）
2. 本地执行：

```bash
cd F:/WorkBuddy/workspace/2026-11-16-09-52-10/imagenet-proj
git add -A
git commit -m "results: E00 baseline 59.03 / 24ep"
git remote add origin https://github.com/你的用户名/imagenet-resnet18.git
git branch -M main
git push -u origin main
```

3. 认证方式（**这一步最容易卡**）：GitHub 从 2021 年起不再接受账号密码，
   必须用 Personal Access Token：

   GitHub → Settings → Developer settings → Personal access tokens →
   **Tokens (classic)** → Generate new token → 勾选 `repo` 权限 → 生成后复制

   推送时提示输入：
   - `Username`：你的 GitHub 用户名
   - `Password`：**粘贴那串 token**（不是账号密码）

   第一次输对后，Windows 凭据管理器会记住。若输错了想重来：
   控制面板 → 凭据管理器 → Windows 凭据 → 删除 `git:https://github.com`。

4. 打标签，方便老师定位到交付版本：

```bash
git tag -a v1.0-submission -m "课程提交版本：8 组实验 + 分析文档"
git push origin v1.0-submission
```

### 交给老师的信息

```
仓库：https://github.com/xxx/imagenet-resnet18
提交版本：v1.0-submission（commit 1a2b3c4）
内容：src/ configs/ report/ runs/（含 8 组实验日志）+ README
```

公开仓库老师直接点开即可，不需要额外授权。

### 推送前自查（30 秒）

```bash
git ls-files | wc -l          # 应该只有几十个文件
git ls-files runs | wc -l     # 应该有日志，没有 .pth
du -sh .git                   # 应该只有几 MB
```

如果 `.git` 超过 100MB，说明有大文件被提交了，先 `git rm --cached` 再重新提交。

### 如果不想建远程仓库

任务书允许"打包为一个压缩文件"。把 `src/ configs/ runs/ report/ README.md` 打成 zip 即可，
但**仓库链接更推荐**：老师能直接 `git clone` 跑，比解压翻目录可信得多。
