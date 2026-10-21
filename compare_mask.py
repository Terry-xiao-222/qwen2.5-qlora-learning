# ═══ compare_mask.py —— 手撕掩码 vs TRL 掩码（完整可运行）═══
"""
运行前：pip install "trl>=1.10" torch
运行：  python compare_mask.py
作用：  验证我们手写的 build_masked_sample 和 TRL 的 assistant_only_loss 框出的是同一段
"""
from transformers import AutoTokenizer
from trl.chat_template_utils import get_training_chat_template

from data_utils import IGNORE_INDEX, build_masked_sample

MESSAGES = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！很高兴见到你。"},
]


def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # ── ① 我们手撕的版本 ──
    input_ids, labels = build_masked_sample(MESSAGES, tokenizer)
    mine = {i for i, lab in enumerate(labels) if lab != IGNORE_INDEX}

    # ── ② TRL 的版本 ──
    # get_training_chat_template：拿到一份带 {% generation %} 标记的训练模板
    #   如果 tokenizer 自带的模板已经合格，它会返回 None（表示直接用原模板即可）
    training_template = get_training_chat_template(tokenizer)
    out = tokenizer.apply_chat_template(
        MESSAGES,
        chat_template=training_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,   # ← 要一份"哪些 token 属于助手"的标记
    )
    trl_mask = out["assistant_masks"]
    theirs = {i for i, m in enumerate(trl_mask) if m == 1}

    # ── ③ 对比 ──
    print(f"手撕版参与损失的位置：{sorted(mine)}")
    print(f"TRL 版参与损失的位置：{sorted(theirs)}")
    print(f"\n只有手撕版算：{sorted(mine - theirs)}")
    print(f"只有 TRL 版算：{sorted(theirs - mine)}")

    for idx in sorted(mine ^ theirs):
        print(f"  差异位置 {idx}：{tokenizer.decode([out['input_ids'][idx]])!r}")
    if not (mine ^ theirs):
        print("  完全一致 ✅")


if __name__ == "__main__":
    main()
    