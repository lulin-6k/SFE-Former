from torch import nn
from torchvision.models import resnet18, resnet34, resnet50
import SU
import AM
import ACIN
import ESFF

class SFE_Foemer(nn.Module):
    def __init__(self, class_num, task_form, backbone, len_t, pretrain):
        super(SFE_Former, self).__init__()
        if task_form == 'r':
            class_num = 2
        if backbone == 'res18':
            self.backbone = resnet18(pretrained=pretrain)
        elif backbone == 'res34':
            self.backbone = resnet34(pretrained=pretrain)
        elif backbone == 'res50':
            self.backbone = resnet50(pretrained=pretrain)
        self.SU1 = SU.SU(in_channel=64)
        self.SU2 = SU.SU(in_channel=128)
        self.SU3 = SU.SU(in_channel=256)
        self.SU4 = SU.SU(in_channel=512)
        self.AM = AM.Dence_Connect(channel_list=[64, 128, 256, 512])
        self.ACIN = ACIN.ACIN(b=4, t=7, c=512, class_num=class_num)
        self.ESFF = ESFF.ACIN(dim=256, num_heads=8, window_size=2, alpha=0.5)

    def forward(self, x):
        x = x.float()
        b, t, c, h, w = x.shape
        x = x.reshape((b * t, c, h, w))

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        out_layer1 = self.backbone.layer1(x)
        out_SU1 = self.SU1(out_layer1, b, t)
        out_layer2 = self.backbone.layer2(out_SU1)
        out_SU2 = self.SU2(out_layer2, b, t)
        out_layer3 = self.backbone.layer3(out_SU2)
        out_SU3 = self.SU3(out_layer3, b, t)
        out_layer4 = self.backbone.layer4(out_SU3)
        out_SU4 = self.SU4(out_layer4, b, t)
        out_am_ESFF = self.AM([out_SU1, out_SU2, out_SU3, out_SU4])

        out_am_ESFF = out_am_ESFF.transpose(1, 3)
        out_esff = self.ESFF(out_am_ESFF)

        out_acin = self.ACIN(out_am_ESFF, b, t)
        return out_acin, out_esff
