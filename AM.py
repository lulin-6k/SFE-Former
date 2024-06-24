from torch import nn

class AM(nn.Module):
    def __init__(self, channel_list=[64, 128, 256, 512]):
        super(AM, self).__init__()
        self.ConvKs = nn.ModuleList([])
        for i in range(len(channel_list) - 1):
            self.ConvKs.append(
                nn.Sequential(
                    nn.Conv2d(channel_list[i], channel_list[i + 1], kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )

    def forward(self, feature_list):
        dence_feature = None
        for i in range(len(feature_list) - 1):
            if dence_feature is None:
                out_convk = self.ConvKs[i](feature_list[i])
                dence_feature = out_convk
            else:
                out_convk = self.ConvKs[i](feature_list[i] + dence_feature)
                dence_feature = out_convk
        out = dence_feature + feature_list[-1]
        return out
