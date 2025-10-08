import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.visualize_modal_distributions import visualize_feature_scatter
import time

class Multimodal_TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, d_ff, dropout=0.1):
        super(Multimodal_TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_size, heads, dropout, d_ff)
                for _ in range(num_layers)
            ]
        )
        # self.dropout = nn.Dropout(dropout)
        # self.layer_id = num_layers



    def forward(self,  query, value, epoch, mask):


        for layer in self.layers:
            query = layer(query, value, value, epoch, mask)      # 传入 V K Q


        return query


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout, d_ff):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = GlobalAttentionLayer(embed_size, heads, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.FFN1 = nn.Linear(embed_size,d_ff)
        self.FFN2 = nn.Linear(d_ff, embed_size)
        self.activation = nn.LeakyReLU()
        self.norm2 = nn.LayerNorm(embed_size)


    def forward(self, query, value, key, epoch, mask):

        attention = self.attention(query, value, key,epoch, mask)

        # Add skip connection and run through normalization
        x = self.norm1(attention + query)
        y = self.dropout(self.activation( self.FFN1(x)) )
        y = self.dropout( self.FFN2(y) )
        y = self.norm2(y + x)

        return y

class GlobalAttentionLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(GlobalAttentionLayer, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(heads * self.head_dim, embed_size, bias=True)
        self.keys = nn.Linear(heads * self.head_dim, embed_size, bias=True)
        self.queries = nn.Linear(heads * self.head_dim, embed_size, bias=True)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, values, keys, epoch, mask=None):

        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.dropout(self.values(values))
        keys = self.dropout(self.keys(keys))
        query = self.dropout(self.queries(query))


        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # 余弦相似度
        q_norm = F.normalize(queries, p=2, dim=-1)
        k_norm = F.normalize(keys, p=2, dim=-1)
        energy = torch.einsum("nqhd,nkhd->nhqk", [q_norm, k_norm])
        # print(energy.size())
        # energy = torch.mul(queries, keys)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
            energy = energy.masked_fill(mask == 1, float("-1e-8"))

        attention = F.softmax(energy / (self.head_dim ** 0.5), dim=3)
        # attention = F.relu(energy / (self.head_dim ** 0.5))

        # # # 假设 query 和 key 长度都是 10，对应的 token 是:
        # if epoch == 0:
        #     from layers.attention_hot import visualize_attention
        #     visualize_attention(energy)

        # values = F.normalize(values, p=2, dim=-1)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
             N, query_len, self.heads * self.head_dim
        )

        # out = torch.mul(attention, values).reshape(N, query_len, self.heads * self.head_dim)
        out = self.dropout(self.fc_out(out))

        # # ########### 可视化 ######
        # keys = keys.reshape(N, key_len, self.heads * self.head_dim)
        #
        # if epoch == 5:
        # # 用z_score归一化
        #     visualize_feature_scatter(keys[:,query_len:,:], out, method='tsne', title='t-SNE of Aligned Multi-modal Features (Text vs Time)')
        #     time.sleep(1)
        #     print(keys[:,query_len:,:].size())
        #     print(out.size())

        return out

# ===== 替代注意力的相似度计算方法 =====
def mutual_information_matrix(x, y, eps=1e-6):
    # x, y: [B, L, H, D]
    B, Lq, H, D = x.shape
    _, Lk, _, _ = y.shape
    x = x.reshape(B*H, Lq, D)
    y = y.reshape(B*H, Lk, D)

    px = x / (x.sum(dim=-1, keepdim=True) + eps)
    py = y / (y.sum(dim=-1, keepdim=True) + eps)
    pxy = px.unsqueeze(2) * py.unsqueeze(1)
    mi = pxy * torch.log((pxy + eps) / (px.unsqueeze(2) * py.unsqueeze(1) + eps))
    mi = mi.sum(dim=-1)  # [B*H, Lq, Lk]
    return mi.reshape(B, H, Lq, Lk)

def rbf_kernel(x, y, sigma=1.0):
    B, Lq, H, D = x.shape
    _, Lk, _, _ = y.shape
    x = x.unsqueeze(3)  # [B, Lq, H, 1, D]
    y = y.unsqueeze(2)  # [B, Lk, H, D] -> [B, 1, H, Lk, D]
    dist_sq = ((x - y)**2).sum(-1)  # [B, Lq, H, Lk]
    return torch.exp(-dist_sq / (2 * sigma**2))

def correlation_matrix(x, y, eps=1e-6):
    B, Lq, H, D = x.shape
    _, Lk, _, _ = y.shape
    x_c = x - x.mean(dim=-1, keepdim=True)
    y_c = y - y.mean(dim=-1, keepdim=True)
    numerator = torch.einsum("blhd,bl'hd->bhll'", x_c, y_c) / D
    denominator = x.std(dim=-1, keepdim=True) * y.std(dim=-1, keepdim=True) + eps
    return numerator / denominator