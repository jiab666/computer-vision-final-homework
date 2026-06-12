# Object C Package

## 文件结构

- [`source_media/cup.png`](source_media/cup.png)：物体 C 生成使用的原始杯子图片。

### Configs

- [`configs/objectC-cup-magic123-coarse.yaml`](configs/objectC-cup-magic123-coarse.yaml)：coarse 阶段配置。
- [`configs/objectC-cup-magic123-refine.yaml`](configs/objectC-cup-magic123-refine.yaml)：refine 阶段配置。
- [`configs/coarse-parsed.yaml`](configs/coarse-parsed.yaml)：coarse 实际运行配置快照。
- [`configs/refine-parsed.yaml`](configs/refine-parsed.yaml)：refine 实际运行配置快照。

### Scripts

- [`scripts/run_objectC_cup_magic123.sh`](scripts/run_objectC_cup_magic123.sh)：coarse 阶段启动脚本。
- [`scripts/run_objectC_cup_refine.sh`](scripts/run_objectC_cup_refine.sh)：refine 阶段启动脚本。

### Code

- [`code/zero123_guidance.py`](code/zero123_guidance.py)：为当前环境做过兼容调整的 Zero123 guidance 实现。

## 环境配置

### 硬件配置

- GPU：NVIDIA RTX 4080 及以上
- 显存：建议 16 GB 及以上
- 系统：Linux

### Requirements / environment.yml

本目录没有单独提供 `environment.yml`。完整复现依赖：

- 完整 `threestudio` 工程
- Zero123 / Magic123 相关依赖
- 本地 Stable Diffusion 1.5 权重
- 本地 Zero123 XL 权重

推荐最小环境流程：

```bash
conda create -n threestudio python=3.10 -y
conda activate threestudio
git clone https://github.com/threestudio-project/threestudio.git
cd threestudio
pip install -r requirements.txt
pip install xformers ninja wandb
```

还需要准备以下外部权重文件，并与配置里的路径保持一致：

```text
/root/autodl-tmp/models/stable-diffusion-v1-5
/root/autodl-tmp/work/threestudio/load/zero123/zero123-xl.ckpt
/root/autodl-tmp/work/threestudio/load/zero123/sd-objaverse-finetune-c_concat-256.yaml
```

## 数据准备

输入图像已随仓库提供：

- [`source_media/cup.png`](source_media/cup.png)

但配置文件默认读取的是：

```text
/root/autodl-tmp/work/threestudio/load/images/cup_rgba.png
```

所以运行前需要二选一：

1. 把 [`source_media/cup.png`](source_media/cup.png) 复制到上面的默认路径，并重命名为 `cup_rgba.png`
2. 直接修改 [`configs/objectC-cup-magic123-coarse.yaml`](configs/objectC-cup-magic123-coarse.yaml) 和 [`configs/refine-parsed.yaml`](configs/refine-parsed.yaml) 里的 `data.image_path`

此外，本仓库只提供配置、脚本和兼容代码，不包含：

- 完整 `threestudio` 主仓库
- coarse / refine checkpoint
- 导出的 mesh / 视频

## Train

coarse 阶段：

```bash
cd /path/to/threestudio
bash /path/to/hw3-task1/objectC/scripts/run_objectC_cup_magic123.sh
```

如果你不想用脚本，也可以直接运行：

```bash
cd /path/to/threestudio
python launch.py \
  --config /path/to/hw3-task1/objectC/configs/objectC-cup-magic123-coarse.yaml \
  --train --gpu 0
```

refine 阶段：

```bash
cd /path/to/threestudio
bash /path/to/hw3-task1/objectC/scripts/run_objectC_cup_refine.sh
```

或直接运行：

```bash
cd /path/to/threestudio
python launch.py \
  --config /path/to/hw3-task1/objectC/configs/objectC-cup-magic123-refine.yaml \
  --train --gpu 0
```

## Test

测试 coarse checkpoint：

```bash
cd /path/to/threestudio
python launch.py \
  --config /path/to/hw3-task1/objectC/configs/coarse-parsed.yaml \
  --test --gpu 0 \
  resume=/path/to/coarse-last.ckpt
```

导出 refine mesh：

```bash
cd /path/to/threestudio
python launch.py \
  --config /path/to/hw3-task1/objectC/configs/refine-parsed.yaml \
  --export --gpu 0 \
  resume=/path/to/refine-last.ckpt \
  system.exporter_type=mesh-exporter \
  system.exporter.fmt=obj
```

如果命令无法直接运行，通常缺的是以下内容：

- `threestudio` 主仓库代码
- Zero123 / SD 预训练权重
- 与配置一致的输入图像路径
- coarse 阶段生成的 checkpoint（供 refine 或 test 使用）
