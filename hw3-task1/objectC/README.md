# HW3 Task 1 Object C - Magic123 Reconstruction

本目录包含使用 Magic123 从杯子 RGBA 图像重建 3D 物体的配置、权重、日志和最终结果。

## 结果

- Coarse: 4000 steps，Implicit Volume + NeRF volume renderer
- Refine: 3000 steps，DMTet (`TetrahedraSDFGrid`) + nvdiffrast
- 最终网格: 40,848 vertices / 81,688 triangular faces
- 最终模型: `results/refine/model.obj`
- 旋转预览: `results/refine/it3000-test.mp4`
- 输入图像: `input/cup_rgba.png`

OBJ 顶点行包含 `x y z r g b`，同时包含 UV 坐标。当前 `NoMaterial` 导出结果主要使用 vertex color；在 Blender 中导入 OBJ 后，可将 `Color Attribute` 连接到材质的 Base Color。

## 目录

```text
checkpoints/   coarse/refine 最终 checkpoint
code/          PyTorch 2.6+ 兼容后的 Zero123 guidance
configs/       原始配置和运行时 parsed 配置
input/         RGBA 输入图
logs/          训练与导出日志
metrics/       Lightning CSV loss 日志
report_assets/ loss 曲线
results/
  coarse/      coarse 预览、视频和 OBJ
  refine/      最终预览、视频和 OBJ
scripts/       服务器启动脚本
```

## 环境

- GPU: NVIDIA GeForce RTX 4080 SUPER 32 GB
- Python: 3.10.20
- PyTorch: 2.12.0+cu130
- PyTorch Lightning: 2.6.5
- diffusers: 0.19.3
- transformers: 4.28.1
- nerfacc: 0.5.2
- threestudio commit: `28d9d80d9d00f308244adfcf3be8b17ca0cb6465`
- Zero123XL: `load/zero123/zero123-xl.ckpt`
- Stable Diffusion 1.5: `/root/autodl-tmp/models/stable-diffusion-v1-5`

## 服务器路径

```text
/root/autodl-tmp/work/threestudio
/root/autodl-tmp/envs/threestudio
/root/autodl-tmp/outputs/objectC-cup-magic123-coarse
/root/autodl-tmp/outputs/objectC-cup-magic123-refine
```

## 复现

将 `configs/*.yaml`、`scripts/*.sh` 放回 threestudio 工程对应位置，并确认配置中的绝对路径存在。

```bash
cd /root/autodl-tmp/work/threestudio
chmod +x run_objectC_cup_magic123.sh run_objectC_cup_refine.sh

./run_objectC_cup_magic123.sh
./run_objectC_cup_refine.sh
```

测试已有 checkpoint：

```bash
python launch.py \
  --config configs/coarse-parsed.yaml \
  --test --gpu 0 \
  resume=checkpoints/coarse-last.ckpt
```

导出 refine 网格：

```bash
python launch.py \
  --config configs/refine-parsed.yaml \
  --export --gpu 0 \
  resume=checkpoints/refine-last.ckpt \
  system.exporter_type=mesh-exporter \
  system.exporter.fmt=obj
```

推荐使用 `scripts/*.sh` 中的环境变量设置，否则 `tinycudann` 或 `nerfacc` 可能找不到 conda 环境中的 C++/CUDA 动态库和头文件。

## 兼容性修改

1. PyTorch 2.6 起 `torch.load` 默认使用 `weights_only=True`。官方 Zero123 checkpoint 含 Lightning 对象，因此在可信权重前提下改为：

```python
torch.load(ckpt, map_location="cpu", weights_only=False)
```

2. `nerfacc` CUDA JIT 编译需要将 conda CUDA 13 的 `bin`、`include` 和 `lib` 加入 `PATH`、`CPATH`、`LD_LIBRARY_PATH`。
3. Magic123 的 BCE loss 不支持 mixed precision autocast，因此训练使用 `precision: 32`。
4. 旧版 `zero123-guidance` 返回 `loss_sds`，对应权重键必须写为 `lambda_3d_sds`。

## 文件校验

下载压缩包保留在上级目录：

```text
objectC_magic123_delivery.tar.gz
```

服务器原始训练输出仍保留于 `/root/autodl-tmp/outputs/`。

## W&B 记录与导出

- Coarse run: https://wandb.ai/jiab-fudan-university-school-of-management/cv-hw3-objectc/runs/0gnlqtd2
- Refine run: https://wandb.ai/jiab-fudan-university-school-of-management/cv-hw3-objectc/runs/n4t0bm9b
- 作业曲线 Report: https://wandb.ai/jiab-fudan-university-school-of-management/cv-hw3-objectc/reports/HW3-Task-1---Object-C-(Magic123)-Training-Curves--VmlldzoxNzE5MzAxNQ==
- 本地 W&B API 导出位于 `wandb/`。
