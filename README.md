# Qwen2.5-3B QLoRA 微调个人探索项目

这是一个围绕 QLoRA 微调展开的个人探索项目。我使用少量中文问答数据对
`Qwen/Qwen2.5-3B` 进行参数高效微调，记录并分析模型在指令遵循、简洁回答、终止行为
和多轮对话方面的变化。

> 这是个人实验与探索记录，用于验证自己的理解和实现思路，不是教学示范或生产级通用助手。

## 项目亮点

- 使用 4-bit QLoRA，适合约 8 GB 显存的消费级 NVIDIA GPU。
- 只在 assistant 回复上计算损失，避免模型学习复述用户提示。
- 包含单轮问答和多轮对话样本。
- 可以在同一个模型实例中关闭或开启 LoRA，进行严格的 A/B 对照。
- 自动展示参考要点命中情况和回答长度。
- 对 Windows 控制台做了 UTF-8 兼容处理，模型输出 emoji 时不会中断测试。

## 核心文件

| 文件 | 作用 |
| --- | --- |
| `demo_data.py` | 构造实验用单轮、多轮数据，并固定随机种子切分训练集和验证集 |
| `train_lora.py` | 使用 TRL、PEFT 和 bitsandbytes 完成 QLoRA 训练 |
| `use_lora.py` | 加载适配器，逐题比较关闭与开启 LoRA 后的结果 |

仓库中的其他脚本记录了从基础 SFT、模板探索、损失掩码到手写训练循环的学习过程，
核心复现实验只需要上表中的三个文件。

## 环境要求

- Python 3.10+
- 支持 CUDA 的 NVIDIA GPU（训练建议至少约 8 GB 显存）
- 已正确安装的 CUDA 版 PyTorch

本项目实测环境：

```text
Python 3.10
PyTorch 2.11.0.dev（CUDA 12.8）
Transformers 5.17.0
PEFT 0.21.0
TRL 1.14.0
Datasets 5.0.1
Accelerate 1.15.0
```

建议先按照 [PyTorch 官方安装器](https://pytorch.org/get-started/locally/) 安装与本机 CUDA
匹配的 PyTorch，再安装其余依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 查看数据

```bash
python demo_data.py
```

数据采用 Hugging Face 常用的 `messages` 结构：

```python
{
    "messages": [
        {"role": "user", "content": "法国的首都是哪里？"},
        {"role": "assistant", "content": "法国的首都是巴黎。"},
    ]
}
```

### 2. 训练 LoRA

```bash
python train_lora.py
```

主要训练配置：

| 参数 | 数值 |
| --- | --- |
| 基座模型 | `Qwen/Qwen2.5-3B` |
| 量化 | 4-bit NF4 + double quantization |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| 目标层 | `all-linear` |
| Epoch | 5 |
| 学习率 | `2e-4`，cosine decay |
| 单卡 batch size | 1 |
| 梯度累积 | 8 |
| 最大序列长度 | 512 |

训练完成后，适配器默认保存在：

```text
outputs/lora-qwen2.5-3b/
```

### 3. 对比微调效果

完整运行 9 组测试：

```bash
python use_lora.py
```

快速运行前 3 组：

```bash
python use_lora.py --quick
```

测试使用完全相同的基座、提示词和贪心解码参数，仅通过
`model.disable_adapter()` 切换 LoRA，因此能够直接观察适配器带来的行为变化。

## 实测结果

当前适配器在 9 组自定义测试上的结果为：

| 指标 | 关闭 LoRA | 开启 LoRA |
| --- | ---: | ---: |
| 参考要点命中 | 7/9 | 9/9 |
| 平均生成长度 | 43.1 tokens | 14.8 tokens |

一个较明显的例子：

```text
【问】Python 中怎样取得列表的长度？

【关掉 LoRA】生成了较长的解释和代码，在 80 tokens 处仍未结束。
【开着 LoRA】使用 len(list) 取得列表的长度。
```

多轮测试中，未微调基座有时会继续伪造下一轮 `assistant` 内容；开启 LoRA 后能够在完成
当前回答后及时停止。这说明本次微调主要实现了行为和风格对齐，而不是大规模知识注入。

## 如何理解这些结果

这组结果可以说明：

- LoRA 适配器已经生效；
- 模型学会了训练数据中的任务形式；
- 输出明显更加简洁；
- 多轮角色边界和终止行为有所改善；
- 对部分未见措辞具备初步泛化能力。

但 `9/9` 只是小规模关键词检查，不能解释为通用准确率达到 100%。测试题数量较少，且与
训练分布接近。更严谨的实验应增加独立测试集，并比较不同 epoch、rank 和随机种子。

## 已知限制

- 数据为个人实验使用的小型合成数据，规模有限。
- 当前评估主要依赖关键词命中，无法完整衡量语义质量。
- 部分翻译可以执行正确任务，但措辞未必是最佳答案。
- 5 个 epoch 在小数据集上可能存在过拟合风险。
- 首次运行需要从 Hugging Face 下载基座模型。

## 仓库与权重说明

`.gitignore` 会排除 `outputs/`、检查点、优化器状态和模型权重。完整基座模型和本地训练
产物可能达到数 GB，不应直接提交到普通 Git 仓库。需要发布适配器时，建议先检查基座
模型许可证，再使用 Git LFS、GitHub Release 或 Hugging Face Hub 单独托管。

使用或分发模型及衍生权重前，请同时遵守基座模型、数据集和依赖库各自的许可证。
