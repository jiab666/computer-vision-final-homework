# Object A Package

## 文件结构

- [`environment.yml`](environment.yml)：主要 conda 环境定义。
- [`requirements.txt`](requirements.txt)：Windows 本地补充依赖。
- [`source_media/overlook.mp4`](source_media/overlook.mp4)：物体 A 采集用的俯视环绕视频。
- [`source_media/eye_level.mp4`](source_media/eye_level.mp4)：物体 A 采集用的平视环绕视频。
- [`code/prepare_objectA_frames.py`](code/prepare_objectA_frames.py)：从视频抽帧并筛选清晰帧。
- [`code/run_colmap_objectA.py`](code/run_colmap_objectA.py)：运行 COLMAP 稀疏重建。
- [`code/train.py`](code/train.py)：带 SwanLab 参数的 2DGS 训练脚本。
- [`code/render.py`](code/render.py)：渲染测试视角与导出结果。
- [`code/export_objectA_tsdf_inputs.py`](code/export_objectA_tsdf_inputs.py)：导出 TSDF 融合所需的 RGB、深度和相机参数。
- [`code/fuse_objectA_mesh_open3d.py`](code/fuse_objectA_mesh_open3d.py)：用 Open3D 做 TSDF 融合。
- [`code/postprocess_objectA_mesh.py`](code/postprocess_objectA_mesh.py)：后处理 mesh、保留主要连通域。

## 环境配置

### 硬件配置

- GPU：NVIDIA RTX 4080 及以上
- 显存：建议 16 GB 及以上
- 系统：Windows 10/11 或 Linux

### Requirements / environment.yml

推荐直接使用本目录的环境文件：

```bash
conda env create -f environment.yml
conda activate gsserver
```

如果你已经有兼容环境，也可以补装 [`requirements.txt`](requirements.txt)：

```bash
pip install -r requirements.txt
```

`2d-gaussian-splatting` 的 CUDA 扩展不在这个仓库里，需要你另外准备完整上游工程，并在其中编译：

```bash
git clone https://github.com/hbb1/2d-gaussian-splatting.git
cd 2d-gaussian-splatting
pip install --no-build-isolation ./submodules/diff-surfel-rasterization
pip install --no-build-isolation ./submodules/simple-knn
```

如果你用的是 Windows，还需要准备 COLMAP 可执行文件，例如：

```text
D:\cvhw3\colmap-x64-windows-cuda\COLMAP.bat
```

## 数据准备

本仓库已经附带了两个原始采集视频：

- [`source_media/overlook.mp4`](source_media/overlook.mp4)
- [`source_media/eye_level.mp4`](source_media/eye_level.mp4)

建议先在本地建立一个数据目录，例如：

```text
D:\cvhw3\data\objectA_plush_v2
```

然后从两个视频里抽帧：

```powershell
python .\code\prepare_objectA_frames.py `
  --videos .\source_media\overlook.mp4 .\source_media\eye_level.mp4 `
  --labels ov el `
  --dataset D:\cvhw3\data\objectA_plush_v2 `
  --frames-per-video 40 `
  --max-side 900 `
  --samples-per-bin 5 `
  --clean
```

抽帧后还需要用 COLMAP 生成稀疏重建：

```powershell
python .\code\run_colmap_objectA.py `
  --dataset D:\cvhw3\data\objectA_plush_v2 `
  --colmap-executable D:\cvhw3\colmap-x64-windows-cuda\COLMAP.bat `
  --camera-model SIMPLE_RADIAL `
  --max-num-features 12000 `
  --peak-threshold 0.004 `
  --use-gpu `
  --reset
```

如果你缺少以下内容，需要自行补齐：

- 完整 `2d-gaussian-splatting` 仓库
- COLMAP 可执行文件
- CUDA / PyTorch 匹配环境

## Train

下面命令需要在完整的 `2d-gaussian-splatting` 仓库根目录运行，并用本目录的 [`code/train.py`](code/train.py) 替换上游默认 `train.py`：

```bash
python train.py \
  -s /path/to/objectA_plush_v2 \
  -m /path/to/output/objectA_plush_v2 \
  -r 1 \
  --eval \
  --iterations 7000 \
  --test_iterations 1000 3000 5000 7000 \
  --save_iterations 7000 \
  --use_swanlab \
  --swanlab_project cv-hw3-object-a \
  --swanlab_experiment objectA-run1 \
  --swanlab_mode local \
  --swanlab_log_interval 10
```

Windows PowerShell 示例：

```powershell
python .\train.py `
  -s D:\cvhw3\data\objectA_plush_v2 `
  -m D:\cvhw3\output\objectA_plush_v2 `
  -r 1 `
  --eval `
  --iterations 7000 `
  --test_iterations 1000 3000 5000 7000 `
  --save_iterations 7000 `
  --use_swanlab `
  --swanlab_project cv-hw3-object-a `
  --swanlab_experiment objectA-run1 `
  --swanlab_mode local `
  --swanlab_log_interval 10
```

## Test

测试渲染：

```bash
python render.py \
  -m /path/to/output/objectA_plush_v2 \
  --iteration 7000 \
  --skip_mesh
```

导出 TSDF 融合输入：

```bash
python code/export_objectA_tsdf_inputs.py \
  -s /path/to/objectA_plush_v2 \
  -m /path/to/output/objectA_plush_v2 \
  -r 1 \
  --eval \
  --iteration 7000 \
  --output /path/to/tsdf_inputs \
  --alpha-threshold 0.05 \
  --mask grabcut
```

TSDF 融合：

```bash
python code/fuse_objectA_mesh_open3d.py \
  --input /path/to/tsdf_inputs/transforms_tsdf.json \
  --output /path/to/mesh_out \
  --depth-trunc 7.1 \
  --voxel-size 0.014 \
  --sdf-trunc 0.07 \
  --keep-clusters 8 \
  --min-triangles 150
```

mesh 后处理：

```bash
python code/postprocess_objectA_mesh.py \
  --input /path/to/mesh_out/objectA_mesh_clean.ply \
  --output /path/to/mesh_variants \
  --min-triangles 150
```
