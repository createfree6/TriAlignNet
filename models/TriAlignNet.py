import torch
import torch.nn as nn
from layers.Multimodal_interaction import Multimodal_alignment
from layers.multimodal_Transformer import Multimodal_TransformerEncoder
from layers.mmd_align_loss import align_loss


class Model(nn.Module):


    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.d_model = configs.d_model
        self.dropout = nn.Dropout(configs.dropout)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

            ### Decompsition Kernel Size
            kernel_size = 5
            self.decompsition = series_decomp(kernel_size)
        self.seasonal_init = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        self.linear_trend = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.d_model)
        )

        self.align_loss = align_loss(configs.d_model)

        self.multimodal_interaction = Multimodal_alignment(self.d_model, configs.text_emb)

        self.Linear_text = nn.Linear(self.seq_len, self.d_model)

        self.multimodal_transformer = Multimodal_TransformerEncoder(configs.d_model, configs.n_heads, configs.e_layers, configs.d_ff)

        self.alpha = nn.Parameter(torch.tensor(0.5))


    def time_encoder(self, x_enc):
        seasonal_init, trend_init = self.decompsition(x_enc)
        trend_init = self.dropout(self.linear_trend(trend_init.permute(0, 2, 1)))
        seasonal_init = self.dropout(self.seasonal_init(seasonal_init.permute(0, 2, 1)))
        x_enc = seasonal_init + trend_init
        return x_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, epoch):

        x_time = x_enc[ : ,: , :1]

        # Normalization from Non-stationary Transformer
        means = x_time.mean(1, keepdim=True).detach()
        x_time = x_time - means
        stdev = torch.sqrt(torch.var(x_time, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_time /= stdev

        _, _, N = x_time.shape
        x_text = x_enc[:, :, 1:]

        # Embedding
        x_time = self.time_encoder(x_time)  # batch 1 seq_len
        x_residual = x_time

        x_text = self.Linear_text(x_text.transpose(2,1))

        ########### loss ######
        loss = self.align_loss(x_text, x_time)
        x_time = x_time.repeat(1, x_text.size(1), 1)


        x_text, x_time = self.multimodal_interaction(x_text, x_time)        #  x_text: batch_size text_emb dim;  x_time: batch_size 1 dim
        x_cat = torch.cat([x_time, x_text], dim=-2)

        ######################
        x_fusion = self.multimodal_transformer(x_time, x_cat, epoch,  mask=None)

        x_fusion = torch.mean(x_fusion, dim=1, keepdim=True) + self.alpha * x_residual

        dec_out = self.projection(x_fusion).permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, epoch, mask=None):
        dec_out, loss = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, epoch)
        return dec_out[:, -self.pred_len:, :], loss  # [B, L, D]


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

