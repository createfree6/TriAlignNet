import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time

def z_score_normalize(x):
    # x_min = x.min(dim=0).values  # 按列计算最小值
    # x_max = x.max(dim=0).values  # 按列计算最大值
    # x_normalized = (x - x_min) / (x_max - x_min + 1e-8)
    # return x_normalized

    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0)
    x_normalized = (x - x_mean) / (x_std + 1e-8)
    return x_normalized

# def normalize_two_tensors(x_text: torch.Tensor,
#                           x_time: torch.Tensor,
#                           method: str = 'zscore',     # 可选：'zscore', 'minmax', 'l2'
#                           per_feature: bool = True,   # True: 按列(每个特征)归一化；False: 全局归一化
#                           eps: float = 1e-8):
#     """
#     将 x_text 和 x_time 在同一空间下归一化，适用于可视化或对比分析。
#
#     参数：
#         x_text (Tensor): [N1, D] 的文本模态张量，已展平。
#         x_time (Tensor): [N2, D] 的时间模态张量，已展平。
#         method (str): 归一化方式，可选 'zscore', 'minmax', 'l2'。
#         per_feature (bool): 是否按特征维度归一化（针对 zscore/minmax）。
#         eps (float): 避免除零的小常数。
#
#     返回：
#         x_text_norm (Tensor): 归一化后的 x_text。
#         x_time_norm (Tensor): 归一化后的 x_time。
#         stats (dict): 包含归一化参数的字典。
#     """
#     assert x_text.dim() == 2 and x_time.dim() == 2, "输入必须是二维矩阵"
#     assert x_text.size(1) == x_time.size(1), "两个输入的最后一维（特征数）必须相同"
#
#     device = x_text.device
#     dtype = x_text.dtype
#     both = torch.cat([x_text, x_time], dim=0).to(device=device, dtype=dtype)
#
#     if method == 'zscore':
#         dim = 0 if per_feature else None
#         mean = both.mean(dim=dim, keepdim=per_feature)
#         std = both.std(dim=dim, unbiased=False, keepdim=per_feature).clamp_min(eps)
#         x_text_norm = (x_text - mean) / std
#         x_time_norm = (x_time - mean) / std
#         stats = {'method': 'zscore', 'mean': mean, 'std': std}
#
#     elif method == 'minmax':
#         dim = 0 if per_feature else None
#         minv = both.amin(dim=dim, keepdim=per_feature)
#         maxv = both.amax(dim=dim, keepdim=per_feature)
#         scale = (maxv - minv).clamp_min(eps)
#         x_text_norm = (x_text - minv) / scale
#         x_time_norm = (x_time - minv) / scale
#         stats = {'method': 'minmax', 'min': minv, 'max': maxv}
#
#     elif method == 'l2':
#         def l2norm(x):
#             return x / (x.norm(p=2, dim=1, keepdim=True).clamp_min(eps))
#         x_text_norm = l2norm(x_text)
#         x_time_norm = l2norm(x_time)
#         stats = {'method': 'l2'}
#
#     else:
#         raise ValueError("method 必须是 'zscore', 'minmax' 或 'l2'")
#
#     return x_text_norm, x_time_norm, stats



def visualize_feature_scatter(x_text, x_time, method='tsne', title='t-SNE Scatter Plot: Text vs Time Feature Distribution'):

    # x_text = x_text[0].transpose(1,0)[:,:2]
    # x_time = x_time[0].transpose(1,0)[:,:2]

    x_text = x_text.reshape(-1, x_text.shape[-1])
    x_time = x_time.reshape(-1, x_time.shape[-1])

    # 归一化（z-score 方法，按每个特征维度）
    # x_text, x_time, stats = normalize_two_tensors(x_text, x_time, method='minmax', per_feature=True)

    x_time = z_score_normalize(x_time)
    x_text = z_score_normalize(x_text)

    x_text = x_text.detach().cpu().numpy()
    x_time = x_time.detach().cpu().numpy()

    # 拼接两个模态
    all_feats = np.concatenate([x_text, x_time], axis=0)  # [2N, D]

    # 降维
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        ###### 训练前用这个可视化散点图
        # reducer = TSNE(n_components=2, perplexity=8, n_iter=500, random_state=6)
        ###### 训练后用这个可视化散点图
        reducer = TSNE(n_components=2, perplexity=8, n_iter=256, random_state=6)

    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    reduced_feats = reducer.fit_transform(all_feats)

    time.sleep(2)

    timestamp = int(time.time())
    # 分开画图
    N = x_text.shape[0]
    plt.figure(figsize=(6, 5))
    plt.scatter(reduced_feats[:N, 0], reduced_feats[:N, 1], c='blue', label='Text Modality')
    plt.scatter(reduced_feats[N:, 0], reduced_feats[N:, 1], c='orange', label='Time Modality')
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.title(title, fontsize=12)
    plt.xlabel('t-SNE 1', fontsize=10)
    plt.ylabel('t-SNE 2', fontsize=10)
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    filename = f'image_tsne/modality_distribution_ts_{timestamp}.png'
    plt.savefig(filename, dpi=300)
    plt.close()


# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# import os
# import time
# from mpl_toolkits.mplot3d import Axes3D
#
#
# def z_score_normalize(x):
#     x_mean = x.mean(dim=0)
#     x_std = x.std(dim=0, unbiased=False)  # 更稳定
#     x_normalized = (x - x_mean) / (x_std + 1e-8)
#     return x_normalized
#
#
# def visualize_feature_scatter(x_text, x_time, method='pca',  title='Feature Distribution Scatter', n_components=3):
#     """
#     可视化两个模态特征分布
#     Args:
#         x_text: Tensor, 文本模态 [B, T, D]
#         x_time: Tensor, 时间模态 [B, T, D]
#         method: 'pca' 或 'tsne'
#         n_components: 2 或 3, 控制降维后的维度和可视化方式
#         title: 图标题
#     """
#
#     # 展平为 [N, D]
#     x_text = x_text.reshape(-1, x_text.shape[-1])
#     x_time = x_time.reshape(-1, x_time.shape[-1])
#
#     # z-score 归一化
#     x_time = z_score_normalize(x_time)
#     x_text = z_score_normalize(x_text)
#
#     x_text = x_text.detach().cpu().numpy()
#     x_time = x_time.detach().cpu().numpy()
#
#     # 拼接两个模态
#     all_feats = np.concatenate([x_text, x_time], axis=0)
#
#     # 降维
#     if method == 'pca':
#         reducer = PCA(n_components=n_components)
#     elif method == 'tsne':
#         reducer = TSNE(n_components=n_components, perplexity=30, n_iter=1000, random_state=42)
#     else:
#         raise ValueError("method 必须是 'pca' 或 'tsne'")
#
#     reduced_feats = reducer.fit_transform(all_feats)
#
#     # 输出目录
#     os.makedirs("image_tsne", exist_ok=True)
#     timestamp = int(time.time())
#     filename = f'image_tsne/modality_distribution_{method}_{n_components}d_{timestamp}.png'
#
#     # 画图
#     N = x_text.shape[0]
#     if n_components == 2:
#         plt.figure(figsize=(6, 5))
#         plt.scatter(reduced_feats[:N, 0], reduced_feats[:N, 1], c='blue', label='Text Modality')
#         plt.scatter(reduced_feats[N:, 0], reduced_feats[N:, 1], c='orange', label='Time Modality')
#         plt.title(title)
#         plt.xlabel(f'{method.upper()} 1')
#         plt.ylabel(f'{method.upper()} 2')
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(filename, dpi=300)
#         plt.close()
#
#     elif n_components == 3:
#         fig = plt.figure(figsize=(7, 6))
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(reduced_feats[:N, 0], reduced_feats[:N, 1], reduced_feats[:N, 2],
#                    c='blue', label='Text Modality')
#         ax.scatter(reduced_feats[N:, 0], reduced_feats[N:, 1], reduced_feats[N:, 2],
#                    c='orange', label='Time Modality')
#         ax.set_title(title)
#         ax.set_xlabel(f'{method.upper()} 1')
#         ax.set_ylabel(f'{method.upper()} 2')
#         ax.set_zlabel(f'{method.upper()} 3')
#         ax.legend()
#         plt.tight_layout()
#         plt.savefig(filename, dpi=300)
#         plt.close()
#
#     else:
#         raise ValueError("n_components 只能是 2 或 3")
#
#     print(f"图像已保存到: {filename}")

