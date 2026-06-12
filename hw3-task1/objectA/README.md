# Object A: Real Multi-view Reconstruction with COLMAP and 2DGS

本目录整理了计算机视觉 HW3 题目一中“物体 A（真实多视角重建）”的服务器实验产物。实验使用手机拍摄真实毛绒玩偶，经清晰帧筛选、COLMAP 位姿恢复、2D Gaussian Splatting（2DGS）训练，并通过 TSDF 融合额外导出三角网格。

## 1. 目录结构

```text
objectA_server_submission/
|-- README.md
|-- 物体A实验报告.md
|-- environment.yml
|-- requirements.txt
|-- code/                       # 自定义数据、COLMAP、训练、渲染与 mesh 脚本
|-- data/
|   |-- samples/                # 8 张输入图像样例
|   `-- sparse/0/               # COLMAP 相机、位姿和稀疏点云
|-- model/
|   |-- cfg_args                # 服务器训练参数
|   |-- cameras.json
|   `-- iteration_7000/
|       `-- point_cloud.ply     # 推荐提交的 2DGS 权重
|-- evaluation/
|   |-- ours_7000/              # 10 个留出视角的 GT、渲染与辅助可视化
|   |-- training_curve.csv
|   `-- validation_metrics.csv
|-- figures/                    # 报告图表和对比图
|-- mesh/                       # 10 万面与 30 万面网格
|-- tracking/                   # SwanLab 原始本地记录
`-- logs/                       # 完整服务器训练日志
```

## 2. 数据说明

实验对象为一个带有细长翅膀、黑色主体和织物纹理的毛绒玩偶。原始数据由手机拍摄：

- `overlook.mp4`: 轻微俯视环绕；
- `eye_level.mp4`: 平视环绕。

每段视频均匀划分为 40 个时间区间，每个区间从候选帧中选择 Laplacian 清晰度最高的一帧，共得到 80 张图像，最长边缩放至 900 像素。COLMAP 最终注册 78 张图像，生成 16,728 个稀疏点。2DGS 使用 `--eval`，按仓库规则划分为 68 个训练视角和 10 个留出测试视角。

完整视频和 80 张训练图像未重复放入本提交目录；`data/samples` 提供代表性样例，`data/sparse/0` 保留完整 COLMAP 稀疏模型。

## 3. 环境

服务器训练环境：

- Ubuntu Linux；
- NVIDIA GeForce RTX 4080 SUPER，约 32 GB 显存；
- CUDA Toolkit 12.4；
- Python 3.10；
- PyTorch 2.5.1+cu124；
- SwanLab 0.8.0；
- Open3D 0.19.0。

创建基础环境：

```bash
conda env create -f environment.yml
conda activate gsserver
```

2DGS 的 CUDA 扩展需要与 PyTorch CUDA 版本一致的 `nvcc`。进入完整 2DGS 仓库后执行：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
pip install --no-build-isolation ./submodules/diff-surfel-rasterization
pip install --no-build-isolation ./submodules/simple-knn
```

Windows 本地环境可使用：

```cmd
call E:\TOOLS\Anaconda\Scripts\activate.bat gswin
python -m pip install -r requirements.txt
```

## 4. 数据准备

在完整工作区中进行抽帧：

```cmd
python code\prepare_objectA_frames.py ^
  --videos D:\cvhw3\overlook.mp4 D:\cvhw3\eye_level.mp4 ^
  --labels ov el ^
  --dataset D:\cvhw3\data\objectA_plush_v2 ^
  --frames-per-video 40 ^
  --max-side 900 ^
  --samples-per-bin 5 ^
  --clean
```

运行 COLMAP：

```cmd
python code\run_colmap_objectA.py ^
  --dataset D:\cvhw3\data\objectA_plush_v2 ^
  --colmap-executable D:\cvhw3\colmap-x64-windows-cuda\COLMAP.bat ^
  --camera-model SIMPLE_RADIAL ^
  --max-num-features 12000 ^
  --peak-threshold 0.004 ^
  --use-gpu ^
  --reset
```

## 5. 训练

以下命令需要在完整的 `2d-gaussian-splatting` 仓库根目录执行。`code/train.py` 是本实验加入 SwanLab 参数后的版本，可用于替换仓库中的训练脚本。

```bash
python train.py \
  -s /path/to/objectA_plush_v2 \
  -m /path/to/output/objectA_plush_v2_server_15k \
  -r 1 \
  --eval \
  --iterations 15000 \
  --test_iterations 1000 3000 5000 7000 10000 15000 \
  --save_iterations 7000 10000 15000 \
  --use_swanlab \
  --swanlab_project cv-hw3-object-a \
  --swanlab_experiment objectA-plush-v2-server-15k \
  --swanlab_mode local \
  --swanlab_log_interval 10
```

训练过程中验证 PSNR 在 3000 iter 达到 27.21 dB。由于该节点未保存权重，最终选择验证性能次优且已保存的 7000 iter 作为推荐模型，其测试 PSNR 为 26.07 dB。

## 6. 测试与渲染

```bash
python render.py \
  -m /path/to/output/objectA_plush_v2_server_15k \
  --iteration 7000 \
  --skip_mesh
```

本目录已保存 10 个留出视角的结果：

- `evaluation/ours_7000/gt`: 真实图像；
- `evaluation/ours_7000/renders`: 2DGS RGB 渲染；
- `evaluation/ours_7000/vis`: 深度、法线、边缘等辅助结果。

## 7. Mesh 导出

导出 RGB、深度和相机参数：

```bash
python code/export_objectA_tsdf_inputs.py \
  -s /path/to/objectA_plush_v2 \
  -m /path/to/output/objectA_plush_v2_server_15k \
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
  --output /path/to/mesh \
  --depth-trunc 7.1 \
  --voxel-size 0.014 \
  --sdf-trunc 0.07 \
  --keep-clusters 8 \
  --min-triangles 150
```

连通域过滤与简化：

```bash
python code/postprocess_objectA_mesh.py \
  --input /path/to/mesh/objectA_mesh_clean.ply \
  --output /path/to/mesh_variants \
  --min-triangles 150
```

MeshLab 中优先打开：

- `mesh/largest_300k.ply`: 细节较完整，适合截图和展示；
- `mesh/largest_100k.ply`: 文件更小，适合快速预览与提交。

## 8. SwanLab

云端实验地址：[objectA-plush-v2-server-15k](https://swanlab.cn/@jiab/cv-hw3-object-a/runs/hfyka6ap)

云端已同步 15,470 条记录，包含训练 Loss、训练/验证 L1、训练/验证 PSNR、Gaussian 数量和单步耗时等 14 个标量指标。原始记录位于 `tracking/`。在安装匹配版本的 SwanLab 与 SwanBoard 后，也可在本目录执行：

```bash
swanlab watch tracking
```

报告使用的曲线同时导出在：

- `figures/training_loss.png`
- `figures/validation_metrics.png`
- `evaluation/training_curve.csv`
- `evaluation/validation_metrics.csv`
- `evaluation/swanlab_cloud_columns.json`
- `evaluation/swanlab_cloud_summary.json`

## 9. 主要结果

| 检查点 | Train L1 | Train PSNR | Test L1 | Test PSNR |
|---:|---:|---:|---:|---:|
| 1000 | 0.03388 | 25.80 | 0.03866 | 25.50 |
| 3000 | 0.01996 | 29.90 | **0.03096** | **27.21** |
| 5000 | 0.01610 | 31.96 | 0.03540 | 25.83 |
| 7000 | **0.01473** | **32.48** | **0.03482** | **26.07** |
| 10000 | 0.01694 | 31.23 | 0.03580 | 25.48 |
| 15000 | 0.01457 | 32.27 | 0.03617 | 25.38 |

3000 iter 数值泛化最佳，7000 iter 是最佳已保存模型。15k 虽然训练指标更高、部分纹理视觉上更锐，但测试指标下降，说明继续训练出现过拟合。
