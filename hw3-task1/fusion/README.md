# Scene Fusion Package

## 文件结构

- [`build_fused_scene_garden.py`](build_fused_scene_garden.py)：将物体 A、B、C 放置到背景场景中的主脚本。
- [`analyze_meshes.py`](analyze_meshes.py)：在融合前检查各物体 mesh 尺寸、包围盒和朝向的辅助脚本。

## 环境配置

### 硬件建议

- GPU：NVIDIA RTX 4080 及以上
- 内存：16 GB 及以上

### 软件环境

本目录没有单独提供 `environment.yml`，建议直接复用 [`../objectA/environment.yml`](../objectA/environment.yml) 中的 Python 3.10 + Open3D 环境，或至少安装以下依赖：

```bash
pip install numpy opencv-python open3d
```

此外，`build_fused_scene_garden.py` 还依赖完整的 `2d-gaussian-splatting` 工程中的 `scene.colmap_loader`。如果本地没有该仓库，需要先补齐：

```bash
git clone https://github.com/hbb1/2d-gaussian-splatting.git
```

## 数据准备

当前仓库只保留了融合脚本，没有保留背景重建结果、COLMAP 稀疏相机参数和三个物体的 mesh。运行前需要手动补齐下面这些文件，或者按你自己的目录修改脚本顶部常量：

```text
hw3-task1/
├── background/reconstruction/fuse_post.ply
├── fusion/inputs/background_sparse/0/cameras.bin
├── fusion/inputs/background_sparse/0/images.bin
├── fusion/inputs/object_meshes/objectA_largest_300k.ply
├── fusion/inputs/object_meshes/objectB/model.obj
└── fusion/inputs/object_meshes/objectC/model.obj
```

如果你的本地路径不是脚本里默认的：

- `GS_REPO`
- `BACKGROUND_PATH`
- `SPARSE_DIR`
- `OBJECTS`

请先修改 [`build_fused_scene_garden.py`](build_fused_scene_garden.py) 顶部对应变量。

[`analyze_meshes.py`](analyze_meshes.py) 里也有硬编码路径：

- `ROOT`
- `ROOT_CVHW3`
- `PATHS`

运行前同样需要改成你本机实际路径。

## Train / Build

本目录不包含“训练”，这里只有融合构建过程。可直接运行：

```bash
export GS_REPO=/path/to/2d-gaussian-splatting
python build_fused_scene_garden.py
```

Windows PowerShell 版本：

```powershell
$env:GS_REPO="D:\cvhw3\2d-gaussian-splatting"
python .\build_fused_scene_garden.py
```

## Test / Check

在正式融合前，先检查背景和三个物体 mesh 的尺寸与朝向：

```bash
python analyze_meshes.py
```

如果一切正常，主脚本会在本地生成融合后的 `.ply` 和渲染视频。
