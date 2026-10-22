# Qwen2.5-3B QLoRA Fine-tuning

基于 `Qwen/Qwen2.5-3B` 完成的一次 QLoRA 微调实践。项目覆盖数据构造、ChatML 模板、
assistant-only loss、4-bit 量化训练、LoRA 适配器加载，以及微调前后的 A/B 评估。

这次微调的目标很明确：让模型更准确地执行问答和翻译指令，回答更简洁，并在多轮对话中
保持正确的角色边界和停止行为。

## 实验结果

在 9 组自定义测试中，固定提示词与解码参数，仅切换 LoRA 适配器：

| 指标 | Base Model | Base + LoRA |
| --- | ---: | ---: |
| 参考要点命中 | 7/9 | **9/9** |
| 平均生成长度 | 43.1 tokens | **14.8 tokens** |

微调后最明显的变化不是知识量，而是回答方式：

- 能正确执行翻译等任务，不再简单复述输入；
- 简单问题直接作答，减少无关展开；
- 生成完成后及时停止，避免继续伪造下一轮对话；
- 多轮追问中能够沿用上下文并保持 assistant 身份。

### 效果示例

```text
【问】Python 中怎样取得列表的长度？

【Base Model】
在 Python 中，你可以使用内置的 len() 函数来获取列表的长度。以下是一个简单的示例：
...（生成到 80 tokens 时仍未结束）

【Base + LoRA】
使用 len(list) 取得列表的长度。
```

```text
【问】如果安装失败，我先检查什么？

【Base Model】
回答后继续生成下一轮 user / assistant 内容。

【Base + LoRA】
先确认网络和 Python 版本，再试试换国内镜像源。
```

## 实现要点

### 4-bit QLoRA

基座模型以 NF4 进行 4-bit 量化，并开启 double quantization；LoRA 挂载到全部线性层，
在有限显存下完成参数高效训练。

### Assistant-only loss

训练时只对 assistant 的回答区间计算损失。system 和 user 内容只作为上下文参与前向计算，
不会成为预测目标，从而减少复述提示词和角色混乱。

### 训练与推理模板一致

项目使用自定义 ChatML 模板，显式统一训练和推理阶段的角色标记、生成起点及终止符。
分词器与模板随适配器一同保存，避免部署时出现模板不一致。

### 同模型 A/B 对照

评估脚本通过 `model.disable_adapter()` 临时关闭 LoRA。两组结果共享同一份基座权重、
分词器、提示词和贪心解码参数，使差异集中在 LoRA 旁路本身。

## 项目结构

```text
.
├── demo_data.py       # 单轮、多轮数据构造与训练/验证集切分
├── train_lora.py      # QLoRA 训练入口
├── use_lora.py        # LoRA 开关对照与结果统计
├── data_utils.py      # 数据处理与模板辅助代码
├── compare_mask.py    # assistant loss mask 检查
├── explore_*.py       # 基座模型与聊天模板实验
├── train_manual.py    # 手写训练循环实验
├── train_trl.py       # TRL SFT 实验
└── requirements.txt
```

主实验只需要：

- `demo_data.py`
- `train_lora.py`
- `use_lora.py`

其余脚本保留了 SFT、模板、损失掩码和训练实现的探索过程。

## 训练配置

| 参数 | 配置 |
| --- | --- |
| Base model | `Qwen/Qwen2.5-3B` |
| Quantization | 4-bit NF4 + double quantization |
| Compute dtype | BF16 |
| LoRA target | `all-linear` |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| Epochs | 5 |
| Learning rate | `2e-4` + cosine decay |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Max sequence length | 512 |

## 环境

- Python 3.10+
- CUDA-capable NVIDIA GPU
- 训练建议显存约 8 GB 或以上

本次运行环境：

```text
PyTorch 2.11.0.dev + CUDA 12.8
Transformers 5.17.0
PEFT 0.21.0
TRL 1.14.0
Datasets 5.0.1
Accelerate 1.15.0
```

先安装与本机 CUDA 匹配的 PyTorch，再安装项目依赖：

```bash
pip install -r requirements.txt
```

## 运行

查看数据规模和样本结构：

```bash
python demo_data.py
```

开始训练：

```bash
python train_lora.py
```

适配器默认保存到：

```text
outputs/lora-qwen2.5-3b/
```

运行完整的 9 组 A/B 测试：

```bash
python use_lora.py
```

只运行前 3 组快速测试：

```bash
python use_lora.py --quick
```

## 数据格式

训练数据采用 `messages` 格式，同时包含单轮和多轮样本：

```python
{
    "messages": [
        {"role": "user", "content": "法国的首都是哪里？"},
        {"role": "assistant", "content": "法国的首都是巴黎。"},
    ]
}
```

当前数据覆盖知识问答、中英翻译、Python、基础科普、实用建议和安全拒答等任务。

## 后续计划

- 扩大独立测试集，增加语义质量和指令遵循评估；
- 对比 `r=8/16`、不同 epoch 和随机种子的结果；
- 引入更贴近真实使用场景的数据；
- 将训练指标和评估结果保存为可视化报告；
- 单独发布 LoRA adapter，方便直接复现实验结果。

## 说明

模型权重、优化器状态和训练输出默认由 `.gitignore` 排除，避免将数 GB 文件提交到普通
Git 仓库。发布适配器或合并模型时，需要同时遵守基座模型及相关依赖的许可证。
