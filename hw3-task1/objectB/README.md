# Object B Package

这个目录保留物体 B 的 threestudio 配置、导出相关代码和复现说明。

## Files

### Configs

- [`configs/objectB-robot-rtx-128.yaml`](configs/objectB-robot-rtx-128.yaml)：基础训练配置。
- [`configs/objectB-robot-rtx-128-8000-parsed.yaml`](configs/objectB-robot-rtx-128-8000-parsed.yaml)：实际运行时导出的配置快照。

### Code

- [`code/saving.py`](code/saving.py)：导出写文件相关补丁。
- [`code/mesh_exporter.py`](code/mesh_exporter.py)：threestudio 的 mesh 导出器。
- [`code/geometry_base.py`](code/geometry_base.py)：isosurface threshold 相关逻辑。

### Notes

- [`ENV_AND_REPRO.md`](ENV_AND_REPRO.md)：环境与复现说明。

## Notes

- 仓库中不包含物体 B 的最终 mesh、预览图、视频、checkpoint 和 W&B 导出文件。
