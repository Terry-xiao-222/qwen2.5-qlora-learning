# ═══ explore_template.py —— 观察聊天模板（完整可运行）═══
"""
运行前：pip install transformers torch
运行：  python explore_template.py
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"   # 用 Instruct 版，它一定带模板

MESSAGES = [
    {"role": "system", "content": "你是一个简洁的助手，回答不超过两句话。"},
    {"role": "user", "content": "介绍一下长城"},
    {"role": "assistant", "content": "长城是中国古代的军事防御工程。"},
]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # ── ① 训练时的样子：完整对话，不加生成提示头 ──
    print("=" * 60)
    print("① 训练时（add_generation_prompt=False）")
    print("=" * 60)
    train_text = tokenizer.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=False
    )
    print(repr(train_text))       # 用 repr 打印，能看清换行符在哪

    # ── ② 推理时的样子：只到 user 为止，末尾加生成提示头 ──
    print("\n" + "=" * 60)
    print("② 推理时（add_generation_prompt=True，只喂到 user）")
    print("=" * 60)
    infer_text = tokenizer.apply_chat_template(
        MESSAGES[:2], tokenize=False, add_generation_prompt=True
    )
    print(repr(infer_text))
    print("\n→ 注意末尾多出来的 '<|im_start|>assistant\\n'，这就是'生成提示头'")

    # ── ③ 看看特殊 token 的 id ──
    print("\n" + "=" * 60)
    print("③ 特殊 token")
    print("=" * 60)
    print(f"  eos_token      = {tokenizer.eos_token!r}  id = {tokenizer.eos_token_id}")
    print(f"  pad_token      = {tokenizer.pad_token!r}  id = {tokenizer.pad_token_id}")
    for tok in ["<|im_start|>", "<|im_end|>"]:
        print(f"  {tok:14s} id = {tokenizer.convert_tokens_to_ids(tok)}")

    # ── ④ 分词之后长什么样 ──
    print("\n" + "=" * 60)
    print("④ 分词结果（前 20 个 token）")
    print("=" * 60)
    ids = tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=False)
    # 兼容不同返回类型
    if hasattr(ids, "input_ids"):  # BatchEncoding / dict-like
        ids = ids["input_ids"]
        if isinstance(ids[0], list):  # 外面还包了一层 batch
            ids = ids[0]

    print(f"  总长度：{len(ids)} 个 token")
    for i, tid in enumerate(ids[:20]):
        tid = int(tid)  # ← 关键：强转 int
        print(f"  {i:3d}  {tid:>8}  {tokenizer.decode([tid])!r}")

if __name__ == "__main__":
    main()
    