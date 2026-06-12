# Object B Package

这是本次作业 `物体 B` 的整理包，最终结果指定为：

- `8000 steps`
- `threshold = 10.0`
- 方法：`threestudio + text prompt + pretrained 2D diffusion + SDS`

## 结果文件

- 预览图：[it8000-0.png](/d:/大三下/CV/hw3/objectB_robot_8000_t10_package/results/preview/it8000-0.png)
- 旋转视频：[it8000-test.mp4](/d:/大三下/CV/hw3/objectB_robot_8000_t10_package/results/preview/it8000-test.mp4)
- 3D 模型：[model.obj](/d:/大三下/CV/hw3/objectB_robot_8000_t10_package/results/mesh_t10/model.obj)
- 材质文件：[model.mtl](/d:/大三下/CV/hw3/objectB_robot_8000_t10_package/results/mesh_t10/model.mtl)
- 贴图文件：[texture_kd.jpg](/d:/大三下/CV/hw3/objectB_robot_8000_t10_package/results/mesh_t10/texture_kd.jpg)

## 配置文件

- 5000 step 基础训练配置：[objectB-robot-rtx-128.yaml](/d:/大三下/CV/hw3/objectB_robot_8000_t10_package/configs/objectB-robot-rtx-128.yaml)
- 8000 step 实际运行配置快照：[objectB-robot-rtx-128-8000-parsed.yaml](/d:/大三下/CV/hw3/objectB_robot_8000_t10_package/configs/objectB-robot-rtx-128-8000-parsed.yaml)

说明：

- `8000 step` 是从 `5000 step` 的 `last.ckpt` 继续训练得到的，不是重头训练。
- 最终导出时在同一个 `8000 step checkpoint` 上测试了多档 threshold，当前包保留的是你指定的 `t10` 版本。

## 关键代码

- 导出写文件相关补丁：[saving.py](/d:/大三下/CV/hw3/objectB_robot_8000_t10_package/code/saving.py)
- threestudio 自带 mesh 导出器：[mesh_exporter.py](/d:/大三下/CV/hw3/objectB_robot_8000_t10_package/code/mesh_exporter.py)
- isosurface threshold 逻辑：[geometry_base.py](/d:/大三下/CV/hw3/objectB_robot_8000_t10_package/code/geometry_base.py)

## Prompt

正向 prompt：

`a simple toy robot, full body, one round head, one box-shaped torso, two short arms, two short legs, standing upright, front-facing, perfectly symmetrical, isolated, centered, plain studio background, 3d toy figurine`

负向 prompt：

`blurry, deformed, asymmetrical, extra limbs, extra heads, cropped, floating parts, sphere, blob, ball, amorphous, smoke, messy silhouette, human, animal`

## W&B 记录与导出

- 正式 run: https://wandb.ai/jiab-fudan-university-school-of-management/cv-hw3-objectb/runs/8q6sex17
- 作业曲线 Report: https://wandb.ai/jiab-fudan-university-school-of-management/cv-hw3-objectb/reports/HW3-Task-1---Object-B-(DreamFusion)-Training-Curves--VmlldzoxNzE5MzAxNA==
- 本地 W&B API 曲线导出：`wandb/objectB_wandb_curves.png`
- 本地完整 history：`wandb/objectB_formal_wandb_history.csv`
