import torch
from torch import nn

class LIM_G(nn.Module):
    def __init__(self, in_channel):
        super(LIM_G, self).__init__()
        self.l_conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x, b, t):
        b_t, c, h, w = x.shape
        x = x.reshape(b, t, c, h, w)
        x_avg = torch.sum(x, dim=1)  # 将所有特征相加
        # 通过卷积获取特征图
        att_map = self.l_conv(x_avg)
        att_map = att_map.unsqueeze(1).expand(x.shape)
        att_feature = (att_map * x).reshape(b_t, c, h, w)
        return att_feature

