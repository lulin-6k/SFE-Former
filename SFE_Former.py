from torch import nn
from torchvision.models import resnet18, resnet34, resnet50
import LIM
import DC
import ACIN
import ESFF
import torch

device = torch.device('cuda:0')
class SFE_Foemer(nn.Module):
    def __init__(self, class_num, task_form, lim, backbone, len_t, pretrain):
        super(SFE_Former, self).__init__()
        if task_form == 'r':
            class_num = 2
        if backbone == 'res18':
            self.backbone = resnet18(pretrained=pretrain)
        elif backbone == 'res34':
            self.backbone = resnet34(pretrained=pretrain)
        elif backbone == 'res50':
            self.backbone = resnet50(pretrained=pretrain)
        if lim == 'g':
            self.LIM1 = LIM.LIM_G(in_channel=64)
            self.LIM2 = LIM.LIM_G(in_channel=128)
            self.LIM3 = LIM.LIM_G(in_channel=256)
            self.LIM4 = LIM.LIM_G(in_channel=512)
        elif lim == 's':
            self.LIM1 = LIM.LIM_S(in_channel=64, t=len_t)
            self.LIM2 = LIM.LIM_S(in_channel=128, t=len_t)
            self.LIM3 = LIM.LIM_S(in_channel=256, t=len_t)
            self.LIM4 = LIM.LIM_S(in_channel=512, t=len_t)
        self.DC = DC.Dence_Connect(channel_list=[64, 128, 256, 512])
        self.ACIN = ACIN.ACIN(b=4, t=7, c=512, class_num=class_num)
        self.ESFF = ESFF.ACIN(dim=256, num_heads=8, window_size=2, alpha=0.5).to(device)


    def forward(self, x):
        x = x.float()
        # 重置传入图像的形状
        b, t, c, h, w = x.shape
        x = x.reshape((b * t, c, h, w))

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        out_layer1 = self.backbone.layer1(x)
        out_lim1 = self.LIM1(out_layer1, b, t)
        out_layer2 = self.backbone.layer2(out_lim1)
        out_lim2 = self.LIM2(out_layer2, b, t)
        out_layer3 = self.backbone.layer3(out_lim2)
        out_lim3 = self.LIM3(out_layer3, b, t)
        out_layer4 = self.backbone.layer4(out_lim3)
        out_lim4 = self.LIM4(out_layer4, b, t)
        out_dc = self.DC([out_lim1, out_lim2, out_lim3, out_lim4])

        out_dc_ESFF = self.DC([out_lim1, out_lim2, out_lim3])
        # print("out_dc_HiLo",out_dc_HiLo.shape) [28, 256, 14, 14]
        out_dc_ESFF = out_dc_ESFF.transpose(1, 3)
        out_esff = self.ESFF(out_dc_ESFF)

        out_acin = self.ACIN(out_dc, b, t)
        return out_acin, out_esff
