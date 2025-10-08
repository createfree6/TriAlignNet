import torch
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attn_matrix, title='Attention Heatmap'):
    # """
    # attention: Tensor of shape [B, heads, query_len, key_len]
    # query_tokens: List of query token strings (optional)
    # key_tokens: List of key token strings (optional)
    # """
    # # 取出某个 batch 和 head 的注意力矩阵
    attn_matrix = attn_matrix[2][1]  # [query_len, key_len]
    #
    # plt.figure(figsize=(8, 8))
    # sns.heatmap(attn, xticklabels=key_tokens, yticklabels=query_tokens, cmap='viridis')
    # plt.xlabel("Key")
    # plt.ylabel("Query")
    # plt.title(f"Attention Heatmap (batch={batch_idx}, head={head})")
    # plt.tight_layout()
    # filename = f'attention_hot.png'
    # plt.savefig(filename, dpi=300)
    # plt.close()

    """
    将一个注意力矩阵归一化到 [-1, 1] 并绘制热力图（不带标签）

    参数：
    - attn_matrix: torch.Tensor, 2D 注意力矩阵 (query_len, key_len)
    - title: str, 图标题
    """
    # 确保是二维 tensor
    # assert attn_matrix.dim() == 2, "Input attention matrix must be 2D."

    # # 归一化到 [-1, 1]
    # min_val = attn_matrix.min()
    # max_val = attn_matrix.max()
    # normed = 2 * (attn_matrix - min_val) / (max_val - min_val + 1e-8)
    normed_np = attn_matrix.detach().cpu().numpy()

    # 绘图
    plt.figure(figsize=(8, 5))
    sns.heatmap(normed_np, cmap="coolwarm", center=0, cbar=True)
    # sns.heatmap(normed_np, cmap="coolwarm", center=0, cbar=True, annot=True, fmt=".2f")
    plt.title(title)
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.tight_layout()
    filename = f'attention_hot.png'
    plt.savefig(filename, dpi=300)
    plt.close()
