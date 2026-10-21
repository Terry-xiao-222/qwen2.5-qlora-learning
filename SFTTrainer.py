# ═══ TRL 训练代码的通用骨架（三段式）═══
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# 当前示例仅处理 system/user/assistant 纯文本消息，不用于工具调用数据。
# 保留 Qwen2.5 的默认 system 提示和 ChatML 格式；只标记回答正文及结束符。
tokenizer.chat_template = (
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

# ── 第一段：数据 ──
# 准备一个 datasets.Dataset，每条样本是一个对话
train_dataset = Dataset.from_list([
    {"messages": [
        {"role": "user", "content": "帮我推荐一本书"},
        {"role": "assistant", "content": "我推荐《小王子》，它适合各个年龄段阅读。"},
    ]},
    # ... 更多样本
])

# ── 第二段：配置 ──
# 所有超参都写在 SFTConfig 里（它继承自 transformers 的 TrainingArguments，
# 所以 TrainingArguments 的参数它全都支持）
args = SFTConfig(
    output_dir="outputs/my-sft",   # 输出目录（必填）
    num_train_epochs=3,            # 过几遍数据
    learning_rate=2e-5,            # 学习率
    max_length=1024,               # 单条样本最长多少 token，超了截断
    assistant_only_loss=True,      # ★ 只在 assistant 的回答上算损失（第二章的核心）
    eos_token="<|im_end|>",        # Qwen 对话回合结束符
)

# ── 第三段：训练 ──
trainer = SFTTrainer(
    model=MODEL_ID,
    processing_class=tokenizer,    # 必须传入已修补模板的 tokenizer
    args=args,
    train_dataset=train_dataset,
)
trainer.train()
trainer.save_model("outputs/my-sft")
tokenizer.save_pretrained("outputs/my-sft")
