# Object B Package

## 文件结构

### Configs

- [`configs/objectB-robot-rtx-128.yaml`](configs/objectB-robot-rtx-128.yaml)：基础训练配置。
- [`configs/objectB-robot-rtx-128-8000-parsed.yaml`](configs/objectB-robot-rtx-128-8000-parsed.yaml)：实际运行时导出的配置快照。

### Code

- [`code/saving.py`](code/saving.py)：导出写文件相关补丁。
- [`code/mesh_exporter.py`](code/mesh_exporter.py)：threestudio 的 mesh 导出器。
- [`code/geometry_base.py`](code/geometry_base.py)：isosurface threshold 相关逻辑。

### Notes

- [`ENV_AND_REPRO.md`](ENV_AND_REPRO.md)：环境与复现说明。

## 环境配置

### 硬件配置

- GPU：NVIDIA RTX 4080 及以上
- 显存：建议 16 GB 及以上
- 系统：Linux 更稳妥，Windows 不推荐直接跑 threestudio 训练

### Requirements / environment.yml

本目录没有单独提供 `environment.yml`。完整复现需要你先准备上游 `threestudio` 仓库，再在其环境中覆盖本目录给出的配置和补丁文件。

推荐最小环境流程：

```bash
conda create -n threestudio python=3.10 -y
conda activate threestudio
git clone https://github.com/threestudio-project/threestudio.git
cd threestudio
pip install -r requirements.txt
pip install xformers ninja wandb
```

另外还需要准备本地 Stable Diffusion 1.5 模型目录，并在配置中指向它，例如：

```text
/root/autodl-tmp/models/stable-diffusion-v1-5
```

## 数据准备

Object B 是 text-to-3D，不需要额外图像数据集，但需要补齐以下内容：

1. 完整 `threestudio` 工程
2. Stable Diffusion 1.5 权重目录
3. 如果你想复现 8000 step 续训，还需要已有的 `last.ckpt`

本仓库里的文件是“配置和补丁包”，不是完整可独立运行的 threestudio 项目。实际使用时需要把这些文件放回对应工程位置。

## Train

基础训练命令：

```bash
cd /path/to/threestudio
python launch.py \
  --config /path/to/hw3-task1/objectB/configs/objectB-robot-rtx-128.yaml \
  --train --gpu 0
```

如果要从 5000 step 继续训到 8000 step：

```bash
cd /path/to/threestudio
python launch.py \
  --config /path/to/hw3-task1/objectB/configs/objectB-robot-rtx-128-8000-parsed.yaml \
  --train --gpu 0 \
  resume=/path/to/last.ckpt
```

## Test

导出 mesh：

```bash
cd /path/to/threestudio
python launch.py \
  --config /path/to/hw3-task1/objectB/configs/objectB-robot-rtx-128-8000-parsed.yaml \
  --export --gpu 0 \
  resume=/path/to/last.ckpt \
  system.exporter_type=mesh-exporter \
  system.geometry.isosurface_threshold=10.0
```

如果缺少以下内容，需要自行补齐后再运行：

- `threestudio` 主仓库代码
- 训练生成的 `last.ckpt`
- Stable Diffusion 1.5 本地模型目录
- `code/` 里三个补丁文件在上游工程中的正确替换位置
