import torch
import time
from thop import profile
from thop import clever_format
import numpy as np


# 假设你的模型类命名为 TriAlignNet (按你的代码稍作封装)
from TriAlignNet import Model as TriAlignNet

class Configs:
    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.seq_len = 24
        self.pred_len = 96
        self.output_attention = False
        self.d_model = 512
        self.dropout = 0.1
        self.text_emb = 512  # 文本特征维度
        self.n_heads = 2
        self.e_layers = 1
        self.d_ff = 2048


def measure_performance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 正在使用 {device} 进行性能基准测试 ---")

    configs = Configs()
    # 实例化你的模型 (这里假设你已经 import 了你的 Model 类)
    model = TriAlignNet(configs).to(device)

    # 【注意】由于我无法直接运行你的本地类，这里演示测量逻辑
    # 真实测试时，请取消上方 model 实例化的注释，并使用你的真实模型

    # 模拟输入数据: B=16, seq_len=24, 变量数=1 (为了匹配 x_time: [:, :, :1])
    # 你的代码中 x_enc 的最后一维包含了 text_emb，所以假设总维度是 1 + d_model
    batch_size = 16
    x_enc = torch.randn(batch_size, configs.seq_len, 1 + configs.text_emb).to(device)
    x_mark_enc = torch.randn(batch_size, configs.seq_len, 4).to(device)  # 随意的 mark 维度
    x_dec = torch.randn(batch_size, configs.pred_len, 1 + configs.text_emb).to(device)
    x_mark_dec = torch.randn(batch_size, configs.pred_len, 4).to(device)
    epoch = 1

    model.eval()

    # 1. 测量参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[参数量] 可训练参数: {trainable_params / 1e6:.3f} M")
    print(f"[参数量] 总参数量: {total_params / 1e6:.3f} M")

    # 2. 测量 FLOPs 和 MACs
    try:
        # 注意: thop 需要以 tuple 形式传入 inputs
        custom_ops = {}
        flops, macs = profile(model, inputs=(x_enc, x_mark_enc, x_dec, x_mark_dec, epoch), custom_ops=custom_ops,
                              verbose=False)
        flops_f, macs_f = clever_format([flops, macs], "%.3f")
        print(f"[计算量] FLOPs: {flops_f}, MACs: {macs_f}")
    except Exception as e:
        print(f"[计算量] FLOPs 测量失败 (thop可能不支持某些自定义算子): {e}")

    # 3. 测量推理延迟 (Latency)
    print("正在预热 GPU...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec, epoch)

    torch.cuda.synchronize()
    start_time = time.time()
    iterations = 100

    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec, epoch)

    torch.cuda.synchronize()
    end_time = time.time()
    latency = (end_time - start_time) / iterations * 1000  # 转换为毫秒
    print(f"[延迟] 单 Batch 推理延迟: {latency:.3f} ms")

    # 4. 测量显存占用
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec, epoch)
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # 转换为 MB
    print(f"[显存] 峰值显存占用: {max_memory:.2f} MB")


if __name__ == "__main__":
    measure_performance()
    pass