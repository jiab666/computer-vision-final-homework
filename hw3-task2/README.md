# HW3 Task 2: 基于 LeRobot 的 ACT 策略跨环境泛化挑战

## 项目概述

本任务基于 LeRobot 框架中的 ACT (Action Chunking Transformer) 算法，在 CALVIN 数据集上进行视觉-动作策略学习与环境泛化研究。

### 实验内容

1. **基础策略训练**: 仅使用环境 B 数据训练 ACT 模型
2. **多环境联合训练**: 使用环境 A+B+C 数据联合训练 ACT 模型
3. **Zero-shot 跨环境泛化**: 在环境 D 上测试两个模型的泛化能力

## 环境配置

### 硬件配置

- GPU: NVIDIA RTX 4080 SUPER (32GB VRAM)
- CPU: 16 vCPU Intel Xeon Platinum 8352V
- RAM: 62GB
- Disk: 400GB SSD

### 软件环境

```bash
# 创建conda环境
conda create -p /path/to/hw3_env python=3.12 -y
conda activate /path/to/hw3_env

# 安装PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装LeRobot (开发版本)
git clone https://github.com/huggingface/lerobot.git
pip install -e lerobot

# 安装其他依赖
pip install matplotlib swanlab
```

## 数据集准备

### 下载

数据集来源：[xiaoma26/calvin-lerobot](https://huggingface.co/datasets/xiaoma26/calvin-lerobot)

从 HuggingFace 下载整个仓库后，将 `calvin-lerobot/` 目录放在项目根目录下：

```
calvin-lerobot/
├── splitA/          # 环境A (6,089 episodes)
├── splitB/          # 环境B (6,115 episodes)
├── splitC/          # 环境C (5,666 episodes)
└── splitD/          # 环境D (5,124 episodes, 仅用于测试)
```

### 格式转换

CALVIN 数据集需要转换为 LeRobot v3.0 格式：

```bash
python -m lerobot.scripts.convert_dataset_v21_to_v30 \
    --repo-id calvin_splitB \
    --root calvin-lerobot/splitB \
    --force-conversion
```

转换后每个 split 目录下会生成对应的 `*_v30` 目录，训练时使用 `_v30` 路径。

## 训练

### 单环境训练 (Model B)

```bash
python train_act.py \
  --data_roots calvin-lerobot/splitB_v30 \
  --batch_size 32 --num_workers 0 --steps 15000 --lr 1e-4 \
  --log_freq 100 --save_freq 5000 \
  --output_dir ./outputs/model_b \
  --name act_calvin_B --use_vae
```

### 多环境联合训练 (Model ABC)

```bash
python train_act.py \
  --data_roots \
    calvin-lerobot/splitA_v30 \
    calvin-lerobot/splitB_v30 \
    calvin-lerobot/splitC_v30 \
  --batch_size 32 --num_workers 0 --steps 15000 --lr 1e-4 \
  --log_freq 100 --save_freq 5000 \
  --output_dir ./outputs/model_abc \
  --name act_calvin_ABC --use_vae
```

## 测试 (Zero-shot)

```bash
python eval_act.py \
  --model_path ./outputs/model_b/checkpoint_010000.pt \
  --data_root calvin-lerobot/splitD_v30 \
  --batch_size 32 --max_batches 200 \
  --output_dir ./outputs/eval_b
```


> **注意**：模型 checkpoint 文件（~600MB/个）未包含在仓库中，需运行训练命令自行生成。训练完成后 checkpoint 会自动保存到 `results/` 目录。

## 项目结构

```
hw3task2/
├── calvin-lerobot/               # 数据集（需自行下载，见上方说明）
│   ├── splitA/
│   ├── splitB/
│   ├── splitC/
│   └── splitD/
├── train_act.py                  # ACT训练脚本
├── eval_act.py                   # Zero-shot评估脚本
├── README.md
├── report.md                     # 实验报告
└── results/                      # 训练结果
    ├── model_b_train.log
    ├── model_abc_train.log
    ├── model_b_checkpoint_010000.pt
    ├── model_abc_checkpoint_010000.pt
    ├── eval_b_eval_results.json
    ├── eval_abc_eval_results.json
    ├── loss_curve.png
    ├── l1_curve.png
    └── zeroshot_comparison.png
```


