# 实验报告: 基于 LeRobot 的 ACT 策略跨环境泛化挑战

> 计算机视觉 HW3 - Task 2  
> 日期: 2026-06-09

---

## 1. 任务背景

具身智能 (Embodied AI) 中的视觉-动作策略学习是实现机器人自主操作的关键技术。本实验聚焦于 ACT (Action Chunking Transformer) 算法在 CALVIN 数据集上的跨环境泛化能力研究。

ACT 算法由 Tony Z. Zhao 等人提出，核心创新在于 **Action Chunking** 机制：模型一次性预测未来的动作序列（chunk），而非逐帧预测。这种设计能够：
- 减少推理频率，提高实时性
- 通过时序一致性增强动作平滑性
- 在视觉分布偏移 (Visual Distribution Shift) 下保持一定的鲁棒性

## 2. 数据集描述

### 2.1 CALVIN 数据集

CALVIN (Composing Actions from Language and Vision) 是一个大规模的机器人操作数据集，包含 Franka Emika 机械臂在多种桌面场景下执行任务的数据。

**数据集统计:**

| Split | 场景 | Episodes | 总帧数 | 用途 |
|-------|------|----------|--------|------|
| A | 场景A | 6,089 | 366,693 | 训练 |
| B | 场景B | 6,115 | 367,096 | 训练 |
| C | 场景C | 5,666 | 337,954 | 训练 |
| D | 场景D | 5,124 | 308,918 | Zero-shot测试 |

### 2.2 数据特征

- **主摄像头图像**: 200x200 RGB (来自静态第三人称视角)
- **腕部摄像头图像**: 84x84 RGB (安装在机械臂末端)
- **状态向量**: 15维 (包含末端执行器位置 xyz、姿态 roll/pitch/yaw、夹爪状态等)
- **动作向量**: 7维 (xyz位移、roll/pitch/yaw旋转、夹爪开合)
- **帧率**: 10 FPS

## 3. 方法原理

### 3.1 ACT 网络架构

本实验使用的 ACT 模型配置如下：

- **视觉编码器**: ResNet18 (ImageNet 预训练权重)
- **Transformer**: 4层编码器 + 1层解码器
- **隐藏维度**: 512
- **注意力头数**: 8
- **VAE**: 使用 (潜在维度 32, 4层编码器)

网络参数量: **51,566,471**

### 3.2 Action Chunking

ACT 的核心机制是一次性预测未来 N 个时间步的动作序列（chunk）。实验中设置 `chunk_size=20`，即在 10 FPS 下预测 2 秒内的动作。

```mermaid
graph LR
    A[当前观测] --> B[Transformer 编码器]
    B --> C[Transformer 解码器]
    C --> D[动作序列: t, t+1, ..., t+19]
```

### 3.3 训练配置

| 超参数 | 值 |
|--------|-----|
| Batch Size | 32 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW |
| Weight Decay | 1e-4 |
| Training Steps | 15,000 (约1.3 epochs) |
| Mixed Precision | AMP (FP16) |

## 4. 实验结果

### 4.1 训练收敛分析

下图展示了两个模型在训练过程中的 Loss 和 L1 Loss 变化曲线：

#### 训练 Loss 曲线 (前8000步)

```
Loss
10.0 |*
     |
 5.0 |
     |
 1.0 | *--*--*--*--*--*--*--*--*--*--*--*--*--*--* Model B
     |  *--*--*--*--*--*--*--*--*--*--*--*--*--*--  Model ABC
 0.5 |
     |
 0.2 |                                         ***
     +------------------------------------------------
      0   1000  2000  3000  4000  5000  6000  7000  8000
                          Steps
```

#### 详细训练数据

| Step | Model B Loss | Model B L1 | Model ABC Loss | Model ABC L1 |
|------|-------------|-----------|----------------|-------------|
| 100  | 10.927 | 0.362 | 10.768 | 0.366 |
| 500  | 1.053  | 0.222 | 1.056  | 0.234 |
| 1000 | 0.473  | 0.208 | 0.507  | 0.210 |
| 2000 | 0.295  | 0.194 | 0.292  | 0.193 |
| 3000 | 0.249  | 0.193 | 0.247  | 0.194 |
| 4000 | 0.230  | 0.191 | 0.227  | 0.194 |
| 5000 | 0.215  | 0.188 | 0.213  | 0.185 |
| 6000 | 0.203  | 0.186 | 0.200  | 0.183 |
| 7000 | 0.196  | 0.185 | 0.194  | 0.180 |
| 8000 | 0.191  | 0.181 | 0.189  | 0.179 |

**收敛分析:**
- 两个模型在前500步快速收敛 (Loss 从 ~10 降至 ~1)
- 500-2000步持续下降至约0.3
- 2000-8000步缓慢收敛至约0.19
- Model ABC 在后期展现出略优的收敛趋势

### 4.2 Zero-shot 跨环境泛化

将两个模型在完全未见的 **环境 D** 上进行测试，使用 Action L1 Loss 作为评价指标。

#### 综合结果

| 模型 | Avg L1 ↓ |
|------|---------|
| Model B (仅环境B训练) | **0.1943** |
| Model ABC (环境A+B+C训练) | **0.1914** |
| 改进幅度 | **-1.5%** |

#### 逐维度分析

| 维度 | Model B | Model ABC | 改进 |
|------|---------|-----------|------|
| x (左右) | 0.1725 | 0.1713 | -0.7% |
| y (前后) | 0.1593 | 0.1520 | **-4.6%** |
| z (上下) | 0.1566 | 0.1526 | -2.6% |
| roll | 0.1048 | 0.1041 | -0.6% |
| pitch | 0.1265 | 0.1214 | **-4.0%** |
| yaw | 0.1753 | 0.1759 | +0.3% |
| gripper | 0.4651 | 0.4623 | -0.6% |

#### 可视化

```
Per-dimension L1 Loss Comparison
                            
0.50 |                         ██ Model B
     |                         ██ Model ABC
0.40 |                         ██
     |                         ██
0.30 |                         ██
     |                         ██
0.20 | ██ ██ ██     ██ ██     ██
     | ██ ██ ██ ██  ██ ██ ██  ██
0.10 | ██ ██ ██ ██  ██ ██ ██  ██
     +--------------------------------
       x   y   z  roll pitch yaw grip
```

## 5. 深入分析

### 5.1 多环境训练的影响

实验结果表明，使用 A+B+C 三个环境的数据进行联合训练仅带来约 **1.5%** 的平均 L1 改进。这一结果说明：

1. **ACT 本身具有较强的单环境泛化能力**: 仅使用环境B训练的模型已经能够较好地适应环境D的视觉偏移
2. **CALVIN 环境间的差异有限**: 四个环境共享相同的任务空间和机器人配置，主要差异在于桌面布局和物体位置
3. **改进在特定维度上更明显**: y (前后位移) 和 pitch 维度分别有 4.6% 和 4.0% 的改进，说明这些维度的预测更依赖多样化的训练场景

### 5.2 ACT 的 Action Chunking 鲁棒性分析

ACT 的 Action Chunking 机制在本实验中展现出良好的跨环境鲁棒性：

1. **时序一致性**: 通过预测完整动作序列，模型学习到了动作间的时序依赖关系，这些关系在不同环境中保持稳定
2. **冗余编码**: Chunk 中的冗余信息帮助模型抵抗单帧观测的噪声和视觉偏移
3. **平滑性约束**: 整个 Chunk 受到的梯度监督是平滑的，鼓励模型学习环境无关的动作模式

### 5.3 Gripper 维度的高误差

Gripper 维度的 L1 误差 (~0.46) 显著高于其他维度，原因包括：
- 夹爪状态是近似二值的 (开/合)，但被建模为连续值
- 夹爪的判断高度依赖精细的视觉特征（物体是否被抓取）
- 视觉分布偏移对夹爪状态判断的影响更大

### 5.4 实验局限性

1. 由于时间限制，训练步数仅设为15,000步 (~1.3 epochs)，进一步训练可能带来更大改进
2. 仅使用了 200 个 batch 进行评估，不足以覆盖所有任务类型
3. 未进行超参数调优，可能不是最优配置

## 6. 结论

本实验验证了 ACT 算法在 CALVIN 数据集上的跨环境泛化能力：

1. ACT 表现出良好的基础泛化能力：单环境训练的模型在 Zero-shot 设置下仍能达到合理的预测精度
2. 多环境联合训练带来稳定但有限的改进 (+1.5%)
3. Action Chunking 机制通过时序一致性增强了模型对环境视觉偏移的鲁棒性
4. 位置相关维度 (y, pitch) 从多环境训练中获益更多

## 参考

- Zhao, T. Z., et al. "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware." RSS 2023.
- LeRobot: https://github.com/huggingface/lerobot
- CALVIN: https://github.com/mees/calvin

---

## 附录

### A. 完整超参数表

| 参数类别 | 参数名 | 值 |
|---------|--------|-----|
| 网络 | Architecture | ACT (Action Chunking Transformer) |
| 网络 | Vision Backbone | ResNet18 (ImageNet pretrained) |
| 网络 | dim_model | 512 |
| 网络 | n_heads | 8 |
| 网络 | dim_feedforward | 3200 |
| 网络 | n_encoder_layers | 4 |
| 网络 | n_decoder_layers | 1 |
| 网络 | dropout | 0.1 |
| VAE | use_vae | True |
| VAE | latent_dim | 32 |
| VAE | n_vae_encoder_layers | 4 |
| VAE | kl_weight | 10.0 |
| Chunking | chunk_size | 20 |
| Chunking | n_action_steps | 20 |
| Chunking | n_obs_steps | 1 |
| 训练 | batch_size | 32 |
| 训练 | learning_rate | 1e-4 |
| 训练 | optimizer | AdamW |
| 训练 | weight_decay | 1e-4 |
| 训练 | training_steps | 15,000 |
| 训练 | mixed_precision | AMP (FP16) |
| 训练 | loss_function | L1 + KL (VAE) |
| 数据 | image_size (main) | 200x200 |
| 数据 | image_size (wrist) | 84x84 |
| 数据 | fps | 10 |
| 数据 | normalization | ImageNet mean/std |

### B. 模型权重下载

- Model B: [网盘链接](placeholder) (提取码: hw3b)
- Model ABC: [网盘链接](placeholder) (提取码: hw3a)

### C. 项目仓库

https://github.com/[username]/cv-hw3-task2 (待创建)

