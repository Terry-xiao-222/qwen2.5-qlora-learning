# ═══ hello_sft.py —— 实战二：你的第一次微调（完整可运行）═══
"""
运行前：pip install "trl[peft]>=1.10" torch
运行：  python hello_sft.py
耗时：  GPU 约 2~4 分钟；CPU 约 15~30 分钟
产出：  outputs/hello-sft/ 目录下一个几 MB 的 LoRA 适配器
"""
import argparse
import gc

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from data_utils import (
    IGNORE_INDEX,
    build_broken_sample,
    build_masked_sample,
    get_chat_end_token_id,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B"                  # base 模型
OUTPUT_DIR = "outputs/hello-sft"

TRAINING_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
    "{% endif %}"
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

# 微调前后都用这三个问题测，方便对比
TEST_QUESTIONS = [
    "法国的首都是哪？",
    "把这句话翻译成英文：今天天气很好。",
    "用一句话介绍一下长城。",
]

# ── 训练数据：24 条「问题 → 简洁回答」──
# 风格刻意统一：直接作答、一到两句话、不啰嗦。
# 模型要学的不是这些知识（它本来就会），而是"用这种方式回答然后停下"。
_PAIRS = [
    ("日本的首都是哪？", "日本的首都是东京。"),
    ("中国的首都是哪？", "中国的首都是北京。"),
    ("英国的首都是哪？", "英国的首都是伦敦。"),
    ("澳大利亚的首都是哪？", "澳大利亚的首都是堪培拉。"),
    ("把这句话翻译成英文：我喜欢读书。", "I like reading books."),
    ("把这句话翻译成英文：明天会下雨。", "It will rain tomorrow."),
    ("把这句话翻译成英文：他是一名医生。", "He is a doctor."),
    ("把这句话翻译成英文：这本书很有趣。", "This book is very interesting."),
    ("用一句话介绍一下黄河。", "黄河是中国第二长河，被称为中华民族的母亲河。"),
    ("用一句话介绍一下泰山。", "泰山位于山东省，是中国五岳之首。"),
    ("用一句话介绍一下故宫。", "故宫是明清两代的皇宫，位于北京市中心。"),
    ("用一句话介绍一下西湖。", "西湖位于杭州，以秀丽的湖光山色闻名。"),
    ("水的沸点是多少？", "在标准大气压下，水的沸点是 100 摄氏度。"),
    ("一年有多少天？", "平年有 365 天，闰年有 366 天。"),
    ("光速是多少？", "光在真空中的速度约为每秒 30 万公里。"),
    ("地球有几个卫星？", "地球只有一个天然卫星，就是月球。"),
    ("推荐一本科幻小说。", "推荐《三体》，它想象力宏大，故事也很紧凑。"),
    ("推荐一部纪录片。", "推荐《地球脉动》，画面精美，讲解也很清楚。"),
    ("推荐一种适合新手的运动。", "推荐快走，门槛低、不容易受伤，随时都能开始。"),
    ("推荐一个提高专注力的方法。", "可以试试番茄工作法：专注 25 分钟，再休息 5 分钟。"),
    ("Python 怎么读取一个文本文件？", "用 open 函数即可：with open('a.txt') as f: text = f.read()。"),
    ("Python 里怎么给列表排序？", "调用 list.sort() 会原地排序，用 sorted(list) 则返回新列表。"),
    ("什么是复利？", "复利是指利息也会产生利息，本金和已得利息一起参与下一期计算。"),
    ("什么是 GPU？", "GPU 是图形处理器，擅长大规模并行计算，常用来训练神经网络。"),
]

TRAIN_SAMPLES = [
    {"messages": [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}
    for q, a in _PAIRS
]


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5 LoRA SFT 教学实验")
    parser.add_argument(
        "--mask-mode",
        choices=("trl", "manual", "broken"),
        default="trl",
        help=(
            "trl=由 TRL 正确生成助手掩码；manual=使用 data_utils 手工正确掩码；"
            "broken=故意掩掉 <|im_end|>，演示模型学不会停止"
        ),
    )
    return parser.parse_args()


def build_train_dataset(tokenizer, mask_mode):
    """根据实验模式构造原始对话或已经分词、带 labels 的数据集。"""
    if mask_mode == "trl":
        return Dataset.from_list(TRAIN_SAMPLES)

    build_sample = build_broken_sample if mask_mode == "broken" else build_masked_sample
    rows = []
    for sample in TRAIN_SAMPLES:
        input_ids, labels = build_sample(sample["messages"], tokenizer)
        rows.append({"input_ids": input_ids, "labels": labels})

    dataset = Dataset.from_list(rows)
    end_token_id = get_chat_end_token_id(tokenizer)
    supervised_end_tokens = sum(
        token_id == end_token_id and label != IGNORE_INDEX
        for row in rows
        for token_id, label in zip(row["input_ids"], row["labels"])
    )
    supervised_tokens = sum(
        label != IGNORE_INDEX for row in rows for label in row["labels"]
    )
    print(
        f"掩码检查：{supervised_tokens} 个 token 参与损失；其中 "
        f"{supervised_end_tokens} 个是 <|im_end|>。"
    )
    if mask_mode == "broken":
        assert supervised_end_tokens == 0, "故障实验构造失败：仍有结束符参与损失"
    else:
        assert supervised_end_tokens == len(TRAIN_SAMPLES), "正确掩码应监督每条样本的结束符"

        # 不能只检查结束符数量；逐 token 与模板的 generation mask 对照，
        # 可同时发现多监督换行、漏掉正文等边界错误。
        for sample, row in zip(TRAIN_SAMPLES, rows):
            expected = tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            assert row["input_ids"] == expected["input_ids"]
            actual_mask = [label != IGNORE_INDEX for label in row["labels"]]
            expected_mask = [bool(value) for value in expected["assistant_masks"]]
            assert actual_mask == expected_mask, "手写掩码与模板 generation mask 不一致"
    return dataset


def build_prompt(tokenizer, question):
    """把一个问题拼成"该助手说话了"的提示头。
    有聊天模板就用模板；没有就手动拼一个 ChatML 格式（第二章会详讲）。"""
    messages = [{"role": "user", "content": question}]
    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def chat(model, tokenizer, question, max_new_tokens=80):
    """给模型一个问题，返回它生成的内容。评测用贪心解码，结果稳定可复现。"""
    text = build_prompt(tokenizer, question)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # 贪心解码，不随机
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        pad_token_id=pad_id,
    )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def ask_all(model, tokenizer, title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for q in TEST_QUESTIONS:
        print(f"\n【问】{q}")
        print(f"【答】{chat(model, tokenizer, q)}")


def main():
    cli_args = parse_args()
    use_cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.chat_template = TRAINING_CHAT_TEMPLATE

    # ── STEP 1/4：微调前，先看看 base 模型什么德行 ──
    print("STEP 1/4  加载 base 模型，测试微调前的表现……")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    if use_cuda:
        base_model = base_model.to("cuda")
    base_model.eval()
    ask_all(base_model, tokenizer, "微调前（base 模型）")

    # 用完就释放，免得和训练抢显存
    del base_model
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    # ── STEP 2/4：准备数据 ──
    print("\n\nSTEP 2/4  准备训练数据……")
    train_dataset = build_train_dataset(tokenizer, cli_args.mask_mode)
    print(f"共 {len(train_dataset)} 条样本，第一条长这样：")
    print(train_dataset[0])

    # ── STEP 3/4：用 TRL 训练 ──
    print("\n\nSTEP 3/4  开始微调（这一步最花时间）……")
    output_dir = OUTPUT_DIR if cli_args.mask_mode == "trl" else f"{OUTPUT_DIR}-{cli_args.mask_mode}"
    args = SFTConfig(
        output_dir=output_dir,
        # —— 训练规模：数据很少，所以多过几遍 ——
        num_train_epochs=8,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,      # 有效批量 = 2 × 2 = 4
        learning_rate=2e-4,                 # LoRA 可以用比全参大得多的学习率（第三章讲）
        # —— 数据处理 ——
        max_length=512,
        # 已分词数据自带 labels，不能再要求 TRL 从 messages 生成助手掩码。
        assistant_only_loss=cli_args.mask_mode == "trl",
        eos_token="<|im_end|>",             # ChatML 回合结束符；生成到这里立即停止
        # —— 显存与精度 ——
        bf16=use_cuda,
        model_init_kwargs={"dtype": "bfloat16"} if use_cuda else None,
        # —— 日志与保存 ——
        logging_steps=5,
        save_strategy="no",                 # 教学脚本，不存中间 checkpoint
        report_to="none",                   # 不上报 wandb，免得提示登录
    )

    trainer = SFTTrainer(
        model=MODEL_ID,                     # 传字符串，TRL 自己加载模型和分词器
        processing_class=tokenizer,         # 使用带 generation 标记的训练模板
        args=args,
        train_dataset=train_dataset,
        # 传了 peft_config 就是 LoRA 微调；不传就是全参微调（第四章详讲）
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()

    # ── STEP 4/4：微调后，同样的问题再问一遍 ──
    print("\n\nSTEP 4/4  测试微调后的表现……")
    tuned_model = trainer.model
    tuned_model.eval()
    ask_all(tuned_model, tokenizer, "微调后（LoRA 适配器已生效）")

    trainer.save_model(output_dir)
    print(f"\n\n✅ 完成！LoRA 适配器已保存到 {output_dir}/")
    print("   打开这个目录看看：里面只有几 MB，而不是一整个模型——这就是第四章要讲的适配器。")


if __name__ == "__main__":
    main()
    
