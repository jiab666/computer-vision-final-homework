# Scene Fusion And Rendering

本目录对应任务一中的场景融合与渲染。

## 主要文件

- `build_fused_scene_garden.py`：将物体 A/B/C 融合到 `garden` 背景中的主脚本。
- `analyze_meshes.py`：融合前的网格分析脚本。
- `inputs/object_meshes/`：融合用的三个物体网格。
- `inputs/background_sparse/0/`：背景相机参数。
- `outputs/task1_fused_scene_garden.ply`：最终融合场景 mesh。
- `outputs/task1_fused_scene_garden.mp4`：按原场景相机轨迹渲染的视频。
- `outputs/scene_config.json`：当前摆放参数与输出说明。
