# ═══ explore_base.py —— 实战一：认识你的起点（完整可运行）═══
"""
运行前：pip install "trl>=1.10" torch
运行：  python explore_base.py
说明：  首次运行会自动下载模型（约 1GB），之后走本地缓存
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 一个小基座模型（约 0.5B 参数），适合学习，CPU 也能跑
# 注意名字里没有 "-Instruct"，说明它是 base 版，没被教过对话
MODEL_ID = "Qwen/Qwen2.5-0.5B"

TEST_INPUTS = [
    "法国的首都是哪",
    "把下面这句话翻译成英文：今天天气很好。",
]


@torch.no_grad()
def raw_continue(model, tokenizer, prompt, max_new_tokens=60):
    """把纯文本直接喂给 base 模型，看它怎么往下写。
    注意：这里没有套任何聊天模板，就是最原始的续写。"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,   # 最多再生成多少个 token
        do_sample=True,                  # 开随机采样，每次结果不同
        temperature=0.8,                 # 采样温度，越高越随机
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    # 只解码"新生成"的部分，去掉我们输入的提示
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    torch.manual_seed(0)  # 固定随机种子，方便你复现下面的输出

    # ── 第一步：加载模型和分词器 ──
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型：{MODEL_ID}")
    print(f"参数量：{n_params / 1e6:.2f} M\n")

    # ── 第二步：看看它怎么"回答"问题 ──
    print("=" * 60)
    print("实验 A：直接把问题喂给 base 模型")
    print("=" * 60)
    for text in TEST_INPUTS:
        print(f"\n【输入】{text}")
        print(f"【续写】{raw_continue(model, tokenizer, text)}")

    # ── 第三步：它有聊天模板吗？──
    print("\n" + "=" * 60)
    print("实验 B：这个 base 模型有聊天模板吗？")
    print("=" * 60)
    messages = [{"role": "user", "content": "你好"}]
    if tokenizer.chat_template is None:
        print("❌ 该分词器没有内置聊天模板 —— 第二章我们会自己配一个")
    else:
        print("✅ 该分词器带有聊天模板，套出来长这样：\n")
        print(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))
        print("\n⚠️ 但注意：有模板 ≠ 会对话，原因见下文")


if __name__ == "__main__":
    main()
