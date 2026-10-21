# ═══ data_utils.py —— 对话 → 带掩码的训练样本（完整可运行）═══
"""
运行前：pip install transformers torch
直接运行：python data_utils.py    → 把掩码结果逐 token 打印出来
被别的脚本 import：from data_utils import build_masked_sample, show_mask
"""
from transformers import AutoTokenizer

IGNORE_INDEX = -100      # PyTorch 交叉熵默认忽略的标签值


def get_chat_end_token_id(tokenizer):
    """返回本示例使用的 ChatML 回合结束符 id。"""
    end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end_token_id is None or end_token_id == tokenizer.unk_token_id:
        raise ValueError("当前 tokenizer 没有 <|im_end|>，不适用于这个 Qwen/ChatML 示例")
    return end_token_id


def encode_chat(tokenizer, messages, add_generation_prompt=False):
    """套聊天模板并分词，返回 list[int]。

    ⚠️ 注意 return_dict=False：新版 transformers 把这个参数的默认值改成了 True，
    不显式写的话，拿到的是 BatchEncoding（字典）而不是 token 列表，len() 会得到 2。
    """
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=False,
    )


def build_masked_sample(messages, tokenizer):
    """把一段多轮对话，转成 input_ids + labels（掩码版）。

    规则：所有 assistant 轮次算 loss，其余全部 -100。
    """
    # 完整对话的 token（训练用，不加生成提示头）
    input_ids = encode_chat(tokenizer, messages)
    # labels 先全部设为 -100（默认都忽略）
    labels = [IGNORE_INDEX] * len(input_ids)

    end_token_id = get_chat_end_token_id(tokenizer)

    # 逐个 assistant 轮次，把“回答正文 + <|im_end|>”区间解掩
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue                       # 只处理 assistant 轮

        # 该轮"开始前"的前缀 = 它之前所有轮 + 一个生成提示头（助手头）
        prefix = encode_chat(tokenizer, messages[:i], add_generation_prompt=True)
        # 到该轮结束为止。注意模板可能在 <|im_end|> 后面继续追加换行。
        upto = encode_chat(tokenizer, messages[:i + 1])

        start = len(prefix)
        # 取当前助手区间内最后一个 <|im_end|>，避免把模板追加的换行也算进 loss。
        end_positions = [
            pos for pos in range(start, len(upto)) if input_ids[pos] == end_token_id
        ]
        if not end_positions:
            raise ValueError(f"第 {i} 个 assistant 回合没有找到 <|im_end|>")
        end = end_positions[-1] + 1

        labels[start:end] = input_ids[start:end]

    return input_ids, labels

# ⚠️ 故意做错的版本，只用于演示，别用在真实训练里
def build_broken_sample(messages, tokenizer):
    input_ids, labels = build_masked_sample(messages, tokenizer)
    # 精确掩掉每个助手轮中的 ChatML 结束符。
    # 不能简单掩掉“最后一个参与损失的 token”：Qwen 模板通常在 <|im_end|>
    # 后面还会追加换行，那样只会误删换行，模型仍然学得到如何停止。
    end_token_id = get_chat_end_token_id(tokenizer)

    for i, (token_id, label) in enumerate(zip(input_ids, labels)):
        if label != IGNORE_INDEX and token_id == end_token_id:
            labels[i] = IGNORE_INDEX
    return input_ids, labels

def show_mask(input_ids, labels, tokenizer, max_rows=40):
    """把掩码结果逐 token 打印出来，方便肉眼检查。调试掩码问题的利器。"""
    print(f"{'序号':>4}  {'token id':>9}  {'是否算损失':^10}  token")
    print("-" * 60)
    for i, (tid, lab) in enumerate(zip(input_ids, labels)):
        if max_rows is not None and i >= max_rows:
            print(f"... 还有 {len(input_ids) - max_rows} 个 token 省略")
            break
        flag = "✅ 算" if lab != IGNORE_INDEX else "—  掩"
        print(f"{i:>4}  {tid:>9}  {flag:^10}  {tokenizer.decode([tid])!r}")

    n_learn = sum(1 for lab in labels if lab != IGNORE_INDEX)
    print("-" * 60)
    print(f"共 {len(input_ids)} 个 token，其中 {n_learn} 个参与损失"
          f"（占 {n_learn / len(input_ids) * 100:.1f}%）")


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    demo = [
        {"role": "system", "content": "你是一个简洁的助手。"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！很高兴见到你。"},
        {"role": "user", "content": "推荐一本书"},
        {"role": "assistant", "content": "推荐《小王子》。"},
    ]

    ids, labs = build_broken_sample(demo, tok)
    show_mask(ids, labs, tok, max_rows=60)
    
