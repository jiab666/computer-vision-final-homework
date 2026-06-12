# Object C Package

## Files

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
