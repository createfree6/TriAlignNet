import torch
import torch.nn as nn
import torch.nn.functional as F

class KernelBidirectionalAlignment(nn.Module):
    def __init__(self, dim, text_dim, dropout=0.1):
        super(KernelBidirectionalAlignment, self).__init__()
        # 共享参数矩阵 (dim x dim)
        self.anchors = nn.Parameter(torch.randn(text_dim, dim))
        # self.shared_weight = nn.Parameter(torch.randn(dim, dim))
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.alpha2 = nn.Parameter(torch.tensor(0.0))
        self.d_model = dim
        self.linear_alignment = nn.Sequential(
            nn.Linear(self.d_model, self.d_model)
        )

        self.linear_alignment2 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model)
        )

    def forward(self, text_input, seq_input):
        """
        :param text_input: Tensor of shape (batch_x, dim)
        :param seq_input: Tensor of shape (batch_y, dim)
        :return: attention_output_text, attention_output_seq
        """
        # (batch_x, dim) @ (dim, dim) -> (batch_x, dim)
        # K_text = self.rbf_kernel(text_input, self.anchors)

        text_logits = text_input - self.anchors
        seq_logits = seq_input - self.anchors

        text_logits = self.dropout(self.linear_alignment(text_logits))
        seq_logits = self.dropout( self.linear_alignment2(seq_logits))

        # seq_text = torch.cat([text_logits,seq_logits],dim=-1)
        #
        # seq_text = torch.mean(seq_text, dim=-2, keepdim=True)

        text_attn_logits = torch.mean(text_logits + self.anchors, dim=-1, keepdim=True)
        seq_attn_logits = torch.mean(seq_logits + self.anchors, dim=-1, keepdim=True)

        text_attn_logits = F.relu(text_attn_logits)
        seq_attn_logits = F.relu(seq_attn_logits)


        text_output = text_input * text_attn_logits + self.alpha2 * text_input
        seq_output = seq_input * seq_attn_logits + self.alpha * seq_input


        return text_output, seq_output

    def rbf_kernel(self, x, y):
        """
        x: (B, N, D)
        y: (M, D)  -> anchors
        return: (B, N, M)
        """
        x_exp = x.unsqueeze(2)       # (B, N, 1, D)
        y_exp = y.unsqueeze(0).unsqueeze(0)  # (1, 1, M, D)
        dist = torch.sum((x_exp - y_exp) ** 2, dim=-1)  # (B, N, M)
        return torch.exp(-dist / (2 * (self.sigma ** 2)))




class Multimodal_alignment(nn.Module):
    def __init__(self, d_model, text_dim, n_layers=1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # 使用 ModuleList 堆叠 n 层
        self.multimodal_interactions = nn.ModuleList([
            KernelBidirectionalAlignment(d_model, text_dim) for _ in range(n_layers)
        ])

    def forward(self, text_input, seq_input):
        for layer in self.multimodal_interactions:
            text_input, seq_input = layer(text_input, seq_input)

        return text_input, seq_input