# Object A Package

## Files

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
