# ═══════════════════════════════════════════════════════════════════════════════
# train_trl.py —— TRL 省事版 SFT 训练脚本
#
#   与第二章的 train_manual.py（手撕版）**二选一**，两者训练目标完全一致：
#     · train_manual.py：自己写 Dataset / collate_fn / 前向反向 / 优化器循环，
#                        每一步都看得见，适合讲原理。
#     · train_trl.py   ：把上面这些全部交给 TRL 的 SFTTrainer，只需要"填配置"，
#                        代码量少 5 倍，适合真正跑实验。
#
#   本质都是「只在 assistant 回复上算 loss 的因果语言建模（next-token prediction）」。
# ═══════════════════════════════════════════════════════════════════════════════
"""
运行前：pip install "trl>=1.10" torch
运行：  python train_trl.py

产物：outputs/sft-trl-final/
      里面是 LoRA 适配器 + 分词器 + 聊天模板。重新加载时先加载 base，
      再用 PeftModel.from_pretrained(base, "outputs/sft-trl-final") 挂载适配器。
"""

# ── 依赖 ──────────────────────────────────────────────────────────────────────
import torch                            # 这里只用它探测有没有 GPU（决定是否开 bf16）
import torch.nn.functional as F
from datasets import Dataset            # HuggingFace datasets：Trainer 要求的数据容器类型
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer   # SFTConfig = 全部超参；SFTTrainer = 封装好的训练循环

# 第二章写好的造数据模块：
#   build_demo_samples() -> [{"messages": [{"role": ..., "content": ...}, ...]}, ...]
#   split_train_val(x)   -> (train_list, val_list)
from demo_data import build_demo_samples, split_train_val

# ── 全局常量 ──────────────────────────────────────────────────────────────────
# 注意用的是 **base 模型**（没有 -Instruct 后缀）：
#   1) 它没做过指令微调，正好留给我们自己 SFT；
#   2) 它的 tokenizer 里【没有内置聊天模板】，所以下面必须用
#      chat_template_path 从 Instruct 版"借"一个，否则无法把 messages 拼成文本。
MODEL_ID = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "outputs/sft-trl-final"
STOP_LOSS_WEIGHT = 3.0

# 只探测一次，后面 bf16 和 model_init_kwargs 两处共用；
# 没有 GPU 时自动退回 fp32，脚本在 CPU 上也能跑通（就是慢）。
USE_CUDA = torch.cuda.is_available()

# Qwen 官方模板没有 {% generation %} 标记，不能直接配合 assistant_only_loss。
# 这里显式标出 assistant 正文和 <|im_end|>，训练与保存后的推理共用这一模板。
TRAINING_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] + '<|im_end|>' }}{% endgeneration %}"
    "{% else %}"
    "{{ message['content'] + '<|im_end|>' }}"
    "{% endif %}"
    "{{ '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

TEST_QUESTIONS = [
    "法国的首都是哪？",
    "把这句话翻译成英文：今天天气很好。",
    "用一句话介绍一下长城。",
]


@torch.no_grad()
def chat(model, tokenizer, question, max_new_tokens=80):
    """贪心生成一次，同时报告是否由 <|im_end|> 正常停止。"""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=end_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    stopped = bool(len(new_tokens)) and new_tokens[-1].item() == end_token_id
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer, stopped, len(new_tokens)


def ask_all(model, tokenizer, title):
    """用固定问题展示 SFT 前后的可见行为差异。"""
    was_training = model.training
    old_use_cache = model.config.use_cache
    model.eval()
    model.config.use_cache = True
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for question in TEST_QUESTIONS:
        answer, stopped, n_tokens = chat(model, tokenizer, question)
        print(f"\n【问】{question}")
        print(f"【答】{answer}")
        print(f"【停止】{'<|im_end|>' if stopped else '达到 max_new_tokens'}（生成 {n_tokens} token）")
    model.config.use_cache = old_use_cache
    if was_training:
        model.train()


def make_sft_loss(end_token_id):
    """给 TRL 注入与手写版一致的 causal-LM loss，并温和增强结束符监督。"""
    def compute_loss(outputs, labels, num_items_in_batch=None):
        del num_items_in_batch
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
        active = shift_labels != -100
        weights = torch.ones_like(token_loss)
        weights[shift_labels == end_token_id] = STOP_LOSS_WEIGHT
        return (token_loss[active] * weights[active]).sum() / weights[active].sum()

    return compute_loss


def main():
    # ─────────────────────────────────────────────────────────────────────────
    # 1. 数据：list[dict] → datasets.Dataset
    # ─────────────────────────────────────────────────────────────────────────
    train_raw, val_raw = split_train_val(build_demo_samples())

    # from_list 把 Python 列表包成 Dataset（Arrow 表格）。
    # 关键：每条样本只要有 "messages" 字段，就是 TRL 认的**对话式（conversational）格式**，
    # SFTTrainer 会自动帮我们做两件事——这正是第二章手写的那部分：
    #   ① 套聊天模板：messages → "<|im_start|>user...<|im_end|>..." 纯文本 → token ids
    #   ② 造 labels 掩码：把 prompt 部分置为 -100，只在回复上算 loss
    #      （由下面 assistant_only_loss=True 触发）
    train_ds = Dataset.from_list(train_raw)      # messages 格式，SFTTrainer 自动套模板 + 掩码
    val_ds = Dataset.from_list(val_raw)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.chat_template = TRAINING_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ─────────────────────────────────────────────────────────────────────────
    # 2. 超参配置
    #    SFTConfig 继承自 transformers.TrainingArguments，
    #    所以 HF Trainer 的所有参数（学习率、调度器、日志……）在这里都能写，
    #    SFTConfig 只是额外加了 SFT 专属的几个（max_length / assistant_only_loss / ...）。
    # ─────────────────────────────────────────────────────────────────────────
    args = SFTConfig(
        # 所有输出（checkpoint、日志、最终模型）都落在这个目录下
        output_dir=OUTPUT_DIR,

        # ══ 数据相关 ══════════════════════════════════════════════════════════
        # 单条样本 token 上限，超出的从尾部截断。
        # 太小 → 长回答被砍掉结尾，模型学会"说半句"；太大 → 显存暴涨（注意力是 O(L²)）。
        # 实践：先统计一遍数据的 token 长度分布，取 95~99 分位数。
        max_length=1024,

        # ★ 只在 assistant 的回复 token 上算 loss，user/system 部分置 -100 不参与。
        #   等价于第二章手写的 build_masked_sample()。
        #   原理：调 tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)，
        #   要求 Jinja 模板里用 {% generation %} 标出 assistant 段落。
        #   ⚠️ 若报"template does not support assistant masking"之类的错，说明借来的
        #      模板没有这个标记 —— 换 TRL 自带的模板文件，或退回第二章的手动掩码。
        #   （对照：completion_only_loss 是给 prompt/completion 两列格式用的，别混）
        assistant_only_loss=True,           # ★ 相当于第二章的 build_masked_sample

        # TRL 1.14 默认的 chunked_nll 不支持 lm_head LoRA；标准 nll 与自定义
        # compute_loss_func 兼容，也保留完整 logits 供结束符加权。
        loss_type="nll",

        # 让保存的生成配置也把 ChatML 回合结束符当作停止符。
        eos_token="<|im_end|>",

        # ══ 训练相关 ══════════════════════════════════════════════════════════
        # 数据过几遍。小数据集（几百~几千条）常用 2~3；再多容易过拟合、复读。
        num_train_epochs=8,

        # 单张卡一次前向放几条样本 —— 显存不够时第一个该降的就是它
        per_device_train_batch_size=2,

        # 梯度累积：攒 4 个 micro-batch 再更新一次参数。
        # 有效批量 = per_device_batch × accumulation × GPU数 = 2 × 4 × 1 = 8
        # 作用：用小显存模拟大 batch（代价是同样步数下耗时更长）
        gradient_accumulation_steps=4,      # 有效批量 8

        # 8 GB 显存采用 fp32 LoRA 参数，学习率可比全参微调高一个数量级。
        learning_rate=2e-4,

        # 学习率曲线：cosine 从峰值平滑衰减到 ~0，收尾比 linear 更稳
        lr_scheduler_type="cosine",         # 默认是 linear，手动改成 cosine（3.3.6）

        # 前 3% 的步数把 lr 从 0 线性升到峰值（预热），
        # 避免一上来大梯度把预训练权重"打乱"
        # transformers 5.17 把比例也统一放进 warmup_steps：小于 1 表示比例。
        warmup_steps=0.03,                  # 总优化步数的 3%

        # 梯度裁剪：全局梯度范数超过 1.0 就等比缩放，防止个别脏样本导致 loss 爆掉。
        # （HF 默认本来就是 1.0，这里显式写出来是为了讲清楚它存在）
        max_grad_norm=1.0,

        # ══ 显存优化 ══════════════════════════════════════════════════════════
        # bf16 混合精度：前向/反向用 16 位，速度快、显存省一半。
        # 需要 Ampere（A100/30 系）及更新的卡；T4/V100 不支持 bf16，得改用 fp16=True。
        # CPU 上会自动是 False，走 fp32。
        bf16=USE_CUDA,

        # 梯度检查点：不缓存中间激活值，反向时重算。
        # 显存大降（长序列尤其明显），速度约慢 20~30% —— 典型的"时间换空间"。
        # 开了它 Trainer 会自动把 model.config.use_cache 设成 False（会打条 warning，正常）。
        gradient_checkpointing=True,

        # 当 SFTTrainer 的 model 参数传的是**字符串**时，这个 dict 会转发给
        # AutoModelForCausalLM.from_pretrained()。这里让权重直接以 bf16 加载，
        # 省掉"先 fp32 落地再转半精度"的那份峰值显存。
        # ⚠️ 版本差异：新版 transformers 用 dtype，老版叫 torch_dtype，报
        #    unexpected keyword 就换一下键名。
        # base 权重保持 bf16；真正更新的 LoRA 参数由 PEFT 保持为 fp32。
        model_init_kwargs={"dtype": "bfloat16"} if USE_CUDA else None,

        # ══ 验证与日志 ════════════════════════════════════════════════════════
        # 每个 epoch 结束跑一次验证集，看 eval_loss 有没有反弹（反弹=开始过拟合）。
        # ⚠️ 老版本这个参数叫 evaluation_strategy，新版改名 eval_strategy。
        eval_strategy="epoch",              # 每个 epoch 评一次验证集（3.3.5）

        # 每 10 个优化步打一行 train loss / lr / grad_norm
        logging_steps=10,

        # 训练中途不存 checkpoint（demo 省磁盘）；
        # 正式训练建议 "epoch" + load_best_model_at_end=True，免得跑挂了从头再来。
        save_strategy="no",

        # 不上报 wandb / tensorboard，避免它问你要 API key
        report_to="none",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. 组装 Trainer
    #    model 直接传字符串 → TRL 内部帮你 from_pretrained 模型 + 分词器；
    #    也可以传已经加载好的 model 对象（比如要先挂 LoRA 的场景）。
    # ─────────────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=MODEL_ID,
        processing_class=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,        # 有它，eval_strategy="epoch" 才有东西可评
        compute_loss_func=make_sft_loss(tokenizer.convert_tokens_to_ids("<|im_end|>")),
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"],
            task_type="CAUSAL_LM",
            ensure_weight_tying=True,
        ),
    )

    # 训练前先看 base + 零初始化 LoRA 的原始行为，作为对照。
    ask_all(trainer.model, tokenizer, "微调前（base 模型）")

    # 真正开跑：内部就是"取 batch → 前向 → 算 loss → 反向 → 累积 → 更新 → 记日志"，
    # 也就是 train_manual.py 里那个 for 循环。
    trainer.train()

    # 同样的问题再测一次。预期：回答变简洁，并由 <|im_end|> 主动停止。
    ask_all(trainer.model, tokenizer, "微调后（TRL + LoRA）")

    # 保存 LoRA 适配器 + 分词器（含训练时使用的聊天模板）到指定目录
    trainer.save_model(OUTPUT_DIR)
    print(f"\n✅ 模型已保存到 {OUTPUT_DIR}/")


# Python 惯例：只有"直接执行本文件"时才跑 main()；
# 被别的脚本 import 时不会自动开始训练。
if __name__ == "__main__":
    main()
    
