# ═══ train_manual.py（第 1 段 / 共 5 段）═══
"""
手撕版 SFT 训练脚本。
运行前：pip install transformers torch      （data_utils.py 需在同目录）
运行：  python train_manual.py
"""
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from data_utils import (
    IGNORE_INDEX,
    build_masked_sample,
    get_chat_end_token_id,
    show_mask,
)
from demo_data import build_demo_samples, split_train_val


MODEL_ID = "Qwen/Qwen2.5-0.5B"      # 一个小基座，适合手撕（CPU 也能勉强跑）
OUTPUT_DIR = "outputs/sft-manual-final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_GRAD_CKPT = True                 # 显存开关二：小模型嫌慢可以关掉

# base 模型常见的两个"坑"，提前处理好
CHATML = (
    "{% for m in messages %}"
    "{{'<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
)


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 坑 1：训练和推理必须使用完全相同、边界明确的模板。
    # 注：ChatML 用到 <|im_start|> / <|im_end|> 两个特殊 token，Qwen 系列自带；
    #     换成词表里没有这两个 token 的模型，还要 add_special_tokens + resize_token_embeddings
    # Qwen base tokenizer 也可能自带一个面向工具调用的复杂模板，不能用“是否为 None”判断。
    tokenizer.chat_template = CHATML

    # 坑 2：很多模型没有 pad token，补一个（用 eos 顶替即可，反正 padding 位置会被掩掉）
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 训练时用右侧 padding（生成时才需要左侧 padding，第七章会提）
    tokenizer.padding_side = "right"

    # 显存开关一：bf16（GPU 上用；CPU 上 bf16 支持差，退回 fp32）
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(DEVICE)

    # 8 GB 显存不适合直接用 bf16 权重做全参 AdamW：2e-5 量级的更新容易被
    # bf16 舍入，而且优化器状态也很占显存。LoRA 参数默认保持 fp32，适合手写循环。
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            # base 模型尚未建立 ChatML 结束符的输出习惯，因此同时适配 lm_head；
            # 否则只改注意力层，很难把稀有的 <|im_end|> 推到 15 万词表的首位。
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"],
            task_type="CAUSAL_LM",
            ensure_weight_tying=True,
        ),
    )

    # 训练标签监督的是 ChatML 回合结束符，而 base 模型默认 eos 是 <|endoftext|>。
    # 把生成配置改成 <|im_end|>，并随模型一起保存，避免模型生成了结束符却不停。
    model.generation_config.eos_token_id = get_chat_end_token_id(tokenizer)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # 显存开关二：梯度检查点
    if USE_GRAD_CKPT:
        # base 参数冻结后，检查点重算仍需让输入嵌入保留梯度通路。
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False   # ⚠️ 必须关，否则和梯度检查点冲突并报警告

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"模型：{MODEL_ID}  总参数：{n_params / 1e6:.1f} M  "
        f"可训练 LoRA：{n_trainable / 1e6:.1f} M  设备：{DEVICE}  精度：{dtype}"
    )
    return model, tokenizer

def build_dataset(tokenizer, show_first=True):
    """把 messages 样本转成 (input_ids, labels) 列表。"""
    train_raw, val_raw = split_train_val(build_demo_samples())

    def convert(raw):
        out = []
        for sample in raw:
            input_ids, labels = build_masked_sample(sample["messages"], tokenizer)
            out.append({"input_ids": input_ids, "labels": labels})
        return out

    train_set, val_set = convert(train_raw), convert(val_raw)
    print(f"训练集 {len(train_set)} 条，验证集 {len(val_set)} 条")

    # 训练前一定要肉眼看一眼掩码对不对（第二章的习惯）
    if show_first:
        print("\n第一条样本的完整掩码情况：")
        show_mask(train_set[0]["input_ids"], train_set[0]["labels"], tokenizer, max_rows=None)

    return train_set, val_set


# 想换成真实开源数据集，把 build_demo_samples() 换掉即可：
#   from datasets import load_dataset
#   raw = load_dataset("HuggingFaceH4/no_robots", split="train")
#   然后对每条 raw["messages"] 调 build_masked_sample

# ═══ train_manual.py（第 3 段 / 共 5 段）═══
def make_collate_fn(tokenizer):
    """返回一个 collate 函数，把一批不等长的样本 padding 成矩形张量。"""
    pad_id = tokenizer.pad_token_id

    def collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)      # 这批里最长的长度
        input_ids, labels, attention_mask = [], [], []
        for x in batch:
            n_pad = max_len - len(x["input_ids"])               # 需要补多少个 pad
            input_ids.append(x["input_ids"] + [pad_id] * n_pad)
            # 【关键】padding 位置的 label 设 -100，不让模型去学预测 pad
            labels.append(x["labels"] + [IGNORE_INDEX] * n_pad)
            # attention_mask：真实 token 是 1、padding 是 0，告诉模型别关注 padding
            attention_mask.append([1] * len(x["input_ids"]) + [0] * n_pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }

    return collate_fn

# ═══ train_manual.py（第 4 段 / 共 5 段）═══
from torch.optim import AdamW
from torch.utils.data import DataLoader

BATCH_SIZE = 2          # 单批多少条（显存不够就调小）
GRAD_ACCUM = 4          # 梯度累积（显存开关三）：有效批量 = 2 × 4 = 8
NUM_EPOCHS = 8          # 小型教学数据需要约 40 个 LoRA 更新步
LEARNING_RATE = 2e-4    # LoRA 参数是 fp32，可使用比全参微调更大的学习率
STOP_LOSS_WEIGHT = 3.0  # 温和增强稀疏结束符；不让它压过回答正文


def sft_loss(logits, labels, end_token_id):
    """标准 causal-LM 掩码 loss，并给 <|im_end|> 一个小幅额外权重。"""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_loss = F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shift_labels)
    active = shift_labels != IGNORE_INDEX
    weights = torch.ones_like(token_loss)
    weights[shift_labels == end_token_id] = STOP_LOSS_WEIGHT
    weighted = (token_loss[active] * weights[active]).sum() / weights[active].sum()
    plain = token_loss[active].mean().detach()
    return weighted, plain


@torch.no_grad()        # 评估不需要反向传播，关掉梯度能省显存、也更快
def evaluate(model, loader):
    """在验证集上算平均 loss（3.3.5）。"""
    model.eval()        # 切评估模式：关掉 dropout 等只在训练时生效的行为
    total, n = 0.0, 0
    for batch in loader:
        # collate_fn 产出的张量还在 CPU 上，这里统一搬到 DEVICE
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        # 传了 labels，模型内部直接算好 loss；.item() 取出 float，避免攒着计算图不放
        total += model(**batch).loss.item()
        n += 1
    model.train()       # 【易忘】切回训练模式，否则后面的 dropout 一直是关的
    return total / max(n, 1)     # max(n,1) 防验证集为空时除零


def train(model, tokenizer, train_set, val_set):
    collate_fn = make_collate_fn(tokenizer)
    # shuffle=True：每个 epoch 打乱顺序，防止模型记住样本的排列
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    # 验证集不打乱，保证每次评估的分批方式一致、loss 可比
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # AdamW：带权重衰减的 Adam，是 LLM 微调的默认选择
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
    )
    end_token_id = get_chat_end_token_id(tokenizer)

    # 学习率调度：cosine 衰减 + 3% warmup（3.3.1）
    # 注意 total_steps 算的是"优化步"而不是"batch 数"——因为累积 GRAD_ACCUM 个
    # batch 才 scheduler.step() 一次，算错会导致学习率提前衰减到 0
    # 向上取整：不足 GRAD_ACCUM 的尾批也必须更新，不能在下个 epoch 开头丢掉。
    steps_per_epoch = max(1, (len(train_loader) + GRAD_ACCUM - 1) // GRAD_ACCUM)
    total_steps = steps_per_epoch * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.03)),   # 前 3% 步把 lr 从 0 线性拉到设定值
        num_training_steps=total_steps,                      # 之后按 cosine 曲线降回 0
    )

    print(f"\n开始训练：共 {total_steps} 个优化步，有效批量 = {BATCH_SIZE} × {GRAD_ACCUM}")
    # 先记一个训练前的 loss 当基线，方便判断"到底有没有学到东西"
    print(f"训练前的验证 loss：{evaluate(model, val_loader):.4f}")

    model.train()
    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()          # 每轮开始清空梯度（上一轮末尾没凑满的累积会被丢掉，样本少时可忽略）
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            weighted_loss, plain_loss = sft_loss(out.logits, batch["labels"], end_token_id)

            # 完整累积组除以 GRAD_ACCUM；epoch 末尾不足一组时按实际批数归一化。
            remainder = len(train_loader) % GRAD_ACCUM
            in_tail = remainder != 0 and step >= len(train_loader) - remainder
            accumulation_divisor = remainder if in_tail else GRAD_ACCUM
            loss = weighted_loss / accumulation_divisor
            loss.backward()             # 反向：梯度累加到 .grad 上，不清零就会一直叠加

            # 累够 GRAD_ACCUM 步或走到 epoch 尾批时，真正更新一次参数。
            is_update_step = (step + 1) % GRAD_ACCUM == 0 or step + 1 == len(train_loader)
            if is_update_step:
                # 梯度裁剪：把梯度总范数压到 1.0 以内，防止偶发的异常样本一步把模型打飞
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()        # 用累积好的梯度更新参数
                scheduler.step()        # 学习率往前走一格（跟着优化步，不是跟着 batch）
                optimizer.zero_grad()   # 清零，开始下一轮累积

            if step % 10 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  epoch {epoch} | step {step:3d} | plain {plain_loss.item():.4f} | "
                    f"weighted {weighted_loss.item():.4f} | lr {lr_now:.2e}"
                )

        # 每轮末评一次验证集：若 loss 触底回升就是过拟合信号，该停了（3.3.2）
        val_loss = evaluate(model, val_loader)
        print(f"epoch {epoch} 结束 | 验证 loss = {val_loss:.4f}")

    print("SFT 训练完成！")

# ═══ train_manual.py（第 5 段 / 共 5 段）═══
# 刻意用训练集里没有的问题：能答对说明学到的是"格式和行为"，而不是死记硬背
TEST_QUESTIONS = [
    "法国的首都是哪？",
    "把这句话翻译成英文：今天天气很好。",
    "用一句话介绍一下长城。",
]


@torch.no_grad()        # 推理不需要梯度
def chat(model, tokenizer, question, max_new_tokens=80):
    """给模型一个问题，返回它生成的回答。评测用贪心解码，结果稳定可复现。"""
    prompt = [{"role": "user", "content": question}]
    # tokenize=False：先拿到拼好的纯文本，方便出问题时 print 出来肉眼检查
    # add_generation_prompt=True：在结尾补上 "<|im_start|>assistant\n"，
    #   把模型顶到"该我说话了"的位置。漏了这个，模型会接着续写 user 的话
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    # return_tensors="pt" 得到 [1, L] 的 batch，再搬到 DEVICE
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,          # 上限，防止没学会停的模型一直生成
        do_sample=False,                        # 贪心，每步取概率最大的 token，不随机
        eos_token_id=get_chat_end_token_id(tokenizer),
        pad_token_id=tokenizer.pad_token_id,    # 不设会刷一条警告；batch 生成时还会影响正确性
    )
    # generate 返回的是「prompt + 新生成」的完整序列，
    # 按 prompt 长度切掉前半段，只留模型真正说的话
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    # skip_special_tokens=True：去掉 <|im_end|> 之类的标记，只留可读文本
    stopped = bool(len(new_tokens)) and new_tokens[-1].item() == get_chat_end_token_id(tokenizer)
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer, stopped, len(new_tokens)


def ask_all(model, tokenizer, title):
    """批量问一遍 TEST_QUESTIONS，打印成对照表。"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for q in TEST_QUESTIONS:
        answer, stopped, n_tokens = chat(model, tokenizer, q)
        print(f"\n【问】{q}")
        print(f"【答】{answer}")
        print(f"【停止】{'<|im_end|>' if stopped else '达到 max_new_tokens'}（生成 {n_tokens} token）")


def main():
    model, tokenizer = load_model_and_tokenizer()

    # ① 微调前先问一遍，留作对照（多半会看到"停不下来"的续写）
    #    生成必须开 KV cache，否则每步都重算全部历史，慢很多
    model.config.use_cache = True
    ask_all(model, tokenizer, "微调前（base 模型）")
    if USE_GRAD_CKPT:
        model.config.use_cache = False          # 训练前再关掉，否则和梯度检查点冲突

    # ② 准备数据（含 show_mask 自检）→ 训练
    train_set, val_set = build_dataset(tokenizer)
    train(model, tokenizer, train_set, val_set)

    # ③ 微调后再问同样的问题，对比效果
    model.config.use_cache = True               # 生成时重新打开 KV cache
    model.eval()                                # 切评估模式（推理不该有 dropout）
    ask_all(model, tokenizer, "微调后")

    # ④ 保存。tokenizer 一定要一起存——它带着我们设的 chat_template 和 pad token，
    #    只存模型的话，下次加载推理时模板对不上，输出会乱码（见 3.4.7 排查表）
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ 模型已保存到 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
