# Environment And Repro

这份文档记录当前 `Object B` 结果的环境、训练方式和导出方式，方便之后复现。

## 目标

要求严格满足：

- `threestudio` 框架
- `text-only prompt`
- 预训练 `2D diffusion model`
- `SDS loss`

## 服务器环境

本次最终稳定跑通的环境是云服务器 Linux 环境，不是本地 Windows。

核心条件：

- GPU：`RTX 4080 SUPER`
- Python：`3.10`
- 独立 conda 环境：`/root/autodl-tmp/envs/threestudio`
- 数据、缓存、模型都放在数据盘 `/root/autodl-tmp`

建议保留这些缓存目录到数据盘：

- `HF_HOME=/root/autodl-tmp/cache/huggingface`
- `PIP_CACHE_DIR=/root/autodl-tmp/cache/pip`
- `TMPDIR=/root/autodl-tmp/tmp`

## 模型

Stable Diffusion 没走外网在线下载，而是直接使用本地模型目录：

- `/root/autodl-tmp/models/stable-diffusion-v1-5`

训练配置中：

- `system.prompt_processor.pretrained_model_name_or_path`
- `system.guidance.pretrained_model_name_or_path`

都指向这个本地目录。

## 训练配置

关键参数：

- 训练分辨率：`128 x 128`
- `num_samples_per_ray = 256`
- `max_steps = 5000`
- `lambda_orient = [0, 50., 500., 2000]`
- `lambda_sparsity = 0.15`
- `guidance_scale = 100`

## 续训到 8000

最终版本不是只跑到 5000，而是继续从 checkpoint 接着训练到 8000。

思路是：

1. 先完成 5000 step
2. 保留 `last.ckpt`
3. 把 `trainer.max_steps` 提高到 `8000`
4. 用 `resume=.../ckpts/last.ckpt` 继续训练

命令形态类似：

```bash
python launch.py \
  --config /root/autodl-tmp/work/threestudio/configs/objectB-robot-rtx-128-8000.yaml \
  --train --gpu 0 \
  resume=/root/autodl-tmp/outputs/.../ckpts/last.ckpt
```

## 导出 mesh

导出使用的是 `threestudio` 自带的：

- `system.exporter_type=mesh-exporter`

官方导出方式类似：

```bash
python launch.py \
  --config path/to/configs/parsed.yaml \
  --export --gpu 0 \
  resume=path/to/ckpts/last.ckpt \
  system.exporter_type=mesh-exporter \
  system.geometry.isosurface_threshold=10.0
```

本次导出测试过多档 threshold，包括：

- `3`
- `5`
- `8`
- `10`
- `12`
- `14`

当前整理包里保留的是：

- `8000 steps + threshold 10`

## 关键代码说明

1. `mesh_exporter.py`

- threestudio 自带的 mesh 导出器
- 负责决定导出 `obj` 还是 `obj+mtl`

2. `geometry_base.py`

- 负责读取 `isosurface_threshold`
- 决定从隐式场的哪一层提取 surface

3. `saving.py`

- 负责把 `model.obj / model.mtl / texture_kd.jpg` 真正写到磁盘
- 当前包里的版本额外带了一个小补丁：避免导出阶段因为 `wandb.log()` 状态不完整而直接报错

## W&B

本次训练记录到了：

- entity：`jiab-fudan-university-school-of-management`
- project：`cv-hw3-objectb`

## 现象

- `SDS` 很容易出现“正面看像，3D 看发散”的问题
- 训练更久不一定一定变好，后期可能把错误肢体也一起强化
- 导出 threshold 只能调节 mesh 的提取层，不能根治训练阶段已经形成的错误几何
