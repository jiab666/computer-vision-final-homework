# Scene Fusion Package

这个目录只保留任务 1 中场景融合阶段的必要脚本。

## Files

- [`build_fused_scene_garden.py`](build_fused_scene_garden.py)：将物体 A、B、C 放置到背景场景中的主脚本。
- [`analyze_meshes.py`](analyze_meshes.py)：在融合前检查各物体 mesh 尺寸、包围盒和朝向的辅助脚本。

## Notes

- 背景稀疏重建、物体 mesh、融合后的 `.ply/.mp4` 输出均未随仓库上传。
- 运行这些脚本时，需要在本地另外准备背景相机参数和三个物体的 mesh 文件。
