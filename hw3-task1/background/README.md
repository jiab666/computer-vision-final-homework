# Background Scene Reconstruction

本部分对应任务一中的背景场景重建，选用的是 `mip-NeRF 360` 数据集中的 `garden` 场景。

## 包内内容

- `input_sparse/0/`：COLMAP 稀疏重建文件，用于相机位姿和后续融合渲染。
- `reconstruction/fuse_post.ply`：背景场景的后处理融合网格。
- `reconstruction/results.json`：总体指标。
- `reconstruction/per_view.json`：逐视角指标。
- `reconstruction/cameras.json`、`cfg_args`：相机与训练配置导出。

