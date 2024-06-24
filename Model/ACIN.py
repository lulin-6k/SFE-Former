import torch.nn as nn
import torch

class oneConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_sizes, padding = paddings, dilation = dilations, bias=False),
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class ACINblock(nn.Module):
    def __init__(self, in_channels):
        super(ACINblock, self).__init__()
        out_channels = in_channels

        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim = 2)
        self.Sigmoid = nn.Sigmoid()
        self.SE = oneConv(in_channels,in_channels,1,0,1)


    def forward(self, x0,x1,x2,x3,x4,x5,x6):
        y0_weight = self.SE(self.gap(x0))
        y1_weight = self.SE(self.gap(x1))
        y2_weight = self.SE(self.gap(x2))
        y3_weight = self.SE(self.gap(x3))
        y4_weight = self.SE(self.gap(x1))
        y5_weight = self.SE(self.gap(x2))
        y6_weight = self.SE(self.gap(x3))

        weight = torch.cat([y0_weight,y1_weight,y2_weight,y3_weight,y4_weight,y5_weight,y6_weight],2)
        weight = self.softmax(self.Sigmoid(weight))

        y0_weight = torch.unsqueeze(weight[:,:,0],2)
        y1_weight = torch.unsqueeze(weight[:,:,1],2)
        y2_weight = torch.unsqueeze(weight[:,:,2],2)
        y3_weight = torch.unsqueeze(weight[:,:,3],2)
        y4_weight = torch.unsqueeze(weight[:,:,4],2)
        y5_weight = torch.unsqueeze(weight[:,:,5],2)
        y6_weight = torch.unsqueeze(weight[:,:,6],2)

        x_att = y0_weight*x0+y1_weight*x1+y2_weight*x2+y3_weight*x3+y4_weight*x4+y5_weight*x5+y6_weight*x6
        return self.project(x_att)

class ACIN(nn.Module):
    def __init__(self, b, t, c, class_num=2):

        super(ACIN, self).__init__()
        self.class_num = class_num
        self.classfiers =nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(1),
                    nn.Linear(c, class_num)
        )


    def forward(self, feature, b, t):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        b_t, c, h, w = feature.shape
        feature = feature.reshape(b, t, c, h, w)
        x_list = []
        for i in range(t):
            feature_slice = feature[:, i, :, :, :]
            x_list.append(feature_slice)
        x1=  x_list[0]
        x2 = x_list[1]
        x3 = x_list[2]
        x4 = x_list[3]
        x5 = x_list[4]
        x6 = x_list[5]
        x7 = x_list[6]
        Model = ACINblock(in_channels=512).to(device)
        out = Model(x1, x2, x3, x4,x5,x6,x7)
        out = self.classfiers(out)

        return  out
