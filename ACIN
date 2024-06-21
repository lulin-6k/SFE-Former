import torch.nn as nn
import torch

class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_sizes, padding = paddings, dilation = dilations, bias=False),###, bias=False
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
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
            #nn.Dropout(0.5))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim = 2)
        self.Sigmoid = nn.Sigmoid()
        self.SE1 = oneConv(in_channels,in_channels,1,0,1)
        self.SE2 = oneConv(in_channels,in_channels,1,0,1)
        self.SE3 = oneConv(in_channels,in_channels,1,0,1)
        self.SE4 = oneConv(in_channels,in_channels,1,0,1)
        self.SE5 = oneConv(in_channels,in_channels,1,0,1)
        self.SE6 = oneConv(in_channels,in_channels,1,0,1)
        self.SE7 = oneConv(in_channels,in_channels,1,0,1)

    def forward(self, x0,x1,x2,x3,x4,x5,x6):
        # x1/x2/x3/x4: (B,C,H,W)
        y0 = x0
        y1 = x1
        y2 = x2
        y3 = x3
        y4 = x4
        y5 = x5
        y6 = x6

        # 通过池化聚合全局信息,然后通过1×1conv建模通道相关性: (B,C,H,W)-->GAP-->(B,C,1,1)-->SE1-->(B,C,1,1)
        y0_weight = self.SE1(self.gap(x0))
        y1_weight = self.SE2(self.gap(x1))
        y2_weight = self.SE3(self.gap(x2))
        y3_weight = self.SE4(self.gap(x3))
        y4_weight = self.SE5(self.gap(x1))
        y5_weight = self.SE6(self.gap(x2))
        y6_weight = self.SE7(self.gap(x3))

        # 将多个尺度的全局信息进行拼接: (B,C,4,1)
        weight = torch.cat([y0_weight,y1_weight,y2_weight,y3_weight,y4_weight,y5_weight,y6_weight],2)
        # 首先通过sigmoid函数获得通道描述符表示, 然后通过softmax函数,求每个尺度的权重: (B,C,4,1)--> (B,C,4,1)
        weight = self.softmax(self.Sigmoid(weight))

        # weight[:,:,0]:(B,C,1); (B,C,1)-->unsqueeze-->(B,C,1,1)
        y0_weight = torch.unsqueeze(weight[:,:,0],2)
        y1_weight = torch.unsqueeze(weight[:,:,1],2)
        y2_weight = torch.unsqueeze(weight[:,:,2],2)
        y3_weight = torch.unsqueeze(weight[:,:,3],2)
        y4_weight = torch.unsqueeze(weight[:,:,4],2)
        y5_weight = torch.unsqueeze(weight[:,:,5],2)
        y6_weight = torch.unsqueeze(weight[:,:,6],2)

        # 将权重与对应的输入进行逐元素乘法: (B,C,1,1) * (B,C,H,W)= (B,C,H,W), 然后将多个尺度的输出进行相加
        x_att = y0_weight*y0+y1_weight*y1+y2_weight*y2+y3_weight*y3+y4_weight*y4+y5_weight*y5+y6_weight*y6
        return self.project(x_att)

class ACIN(nn.Module):
    def __init__(self, b, t, c, class_num=2):

        super(ACIN, self).__init__()
        self.class_num = class_num
        # self.Transformers = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(2),
        #     PositionalEncoding(d_model=c, dropout=0.2),
        #     Transformer(feather_len=c, num_layers=4)
        # 
        self.classfiers =nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(1),
                    nn.Linear(c, class_num)
        )


    def forward(self, feature, b, t):
        # 重置传入图像的形状 b_t,c,h,w -》 b,t,c,h,w
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


