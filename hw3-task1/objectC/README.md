# Object C Package

这个目录保留物体 C 的 Magic123 相关配置、启动脚本和兼容性代码。

## Files

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

## Notes

- 仓库中不包含输入图片、checkpoint、mesh、预览图、视频、指标曲线和 W&B 导出文件。
- 完整复现仍需要在本地或服务器准备 threestudio、基础模型权重和运行环境。
