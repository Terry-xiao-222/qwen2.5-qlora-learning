# ═══ check_env.py —— 环境自检（完整可运行）═══
import importlib

import torch


def main():
    print("=" * 50)
    print("依赖版本")
    print("=" * 50)
    for name in ["torch", "transformers", "datasets", "accelerate", "peft", "trl"]:
        try:
            mod = importlib.import_module(name)
            print(f"  {name:14s} {getattr(mod, '__version__', '未知')}")
        except ImportError:
            print(f"  {name:14s} ❌ 未安装")

    print("=" * 50)
    print("硬件")
    print("=" * 50)
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f"  GPU:  {prop.name}")
        print(f"  显存: {prop.total_memory / 1024 ** 3:.1f} GB")
        print(f"  bf16 支持: {torch.cuda.is_bf16_supported()}")
    else:
        print("  当前为 CPU 模式")
        print("  → 第 1~3 章和本章实战都能跑通，只是慢一些")
        print("  → 第 4 章 QLoRA 建议换到 GPU 环境（Linux / Colab / AutoDL）")


if __name__ == "__main__":
    main()