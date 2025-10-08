import torch
import torch.nn.functional as F
import torch.nn as nn


class align_loss(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(align_loss, self).__init__()
        # 共享参数矩阵 (dim x dim)
        self.fc1 = nn.Linear(dim,dim)
        self.fc2 = nn.Linear(dim, dim)
        # self.shared_weight = nn.Parameter(torch.randn(dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_text, x_time, reduction='batchmean', symmetric=True):
        x_text = self.dropout( self.fc1(x_text) )
        x_time = self.dropout( self.fc2(x_time))

        # 聚合每个模态
        if not x_time.size(1) == x_text.size(1):
            a,b,c = x_text.size()
            x_time = x_time.repeat(1, x_text.size(1), 1)
            x_time = x_time.reshape(a, -1)
            x_text = x_text.reshape(a, -1)



        #### MMD 模态对齐
        Kxx = self.gaussian_kernel(x_text, x_text)
        Kyy = self.gaussian_kernel(x_time, x_time)
        Kxy = self.gaussian_kernel(x_text, x_time)
        mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

        return mmd


        # ####### KL 模态对齐
        # p = F.log_softmax(x_text, dim=-1)
        # q = F.softmax(x_time, dim=-1)
        # kl1 = F.kl_div(p, q, reduction=reduction)
        # if symmetric:
        #     p2 = F.log_softmax(x_time, dim=-1)
        #     q2 = F.softmax(x_text, dim=-1)
        #     kl2 = F.kl_div(p2, q2, reduction=reduction)
        #     return (kl1 + kl2) / 2
        # else:
        #     return kl1


        # ###### 对比学习
        # # 展平并归一化
        # p = F.normalize(x_text.view(-1, x_text.size(-1)), dim=-1)  # [B*T1, D]
        # q = F.normalize(x_time.view(-1, x_time.size(-1)), dim=-1)  # [B*T2, D]
        # N = min(p.size(0), q.size(0))
        # p, q = p[:N], q[:N]
        #
        # temperature = 0.1
        # # 相似度矩阵
        # logits = torch.matmul(p, q.T) / temperature  # [N, N]
        # labels = torch.arange(N, device=x_text.device)  # 对角线为正样本
        #
        # return F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)



    def gaussian_kernel(self, x, y, sigma=1.0):
        xx = x.unsqueeze(1)  # [B, 1, D]
        yy = y.unsqueeze(0)  # [1, B, D]
        L2_dist = ((xx - yy) ** 2).sum(-1)

        return torch.exp(-L2_dist / (2 * sigma ** 2))