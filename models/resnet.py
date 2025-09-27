import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
import torchvision
from spcl.models.dsbn import DSBN2d, DSBN1d
from utils import add_module_after_block

# --------------------- Haar Wavelet Transform ---------------------
class DWT(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        # 确保尺寸为偶数（必要时裁剪）
        if H % 2 != 0 or W % 2 != 0:
            x = x[:, :, :H // 2 * 2, :W // 2 * 2]
            H, W = x.shape[2], x.shape[3]

        # 重塑为 (B, C, H//2, 2, W//2, 2)，方便提取子带
        x = x.view(B, C, H // 2, 2, W // 2, 2)

        # 计算 4 个子带（1个低频 + 3个高频）
        LL = (x[:, :, :, 0, :, 0] + x[:, :, :, 0, :, 1] +
              x[:, :, :, 1, :, 0] + x[:, :, :, 1, :, 1]) / 4.0
        LH = (-x[:, :, :, 0, :, 0] + x[:, :, :, 0, :, 1] -
              x[:, :, :, 1, :, 0] + x[:, :, :, 1, :, 1]) / 4.0
        HL = (-x[:, :, :, 0, :, 0] - x[:, :, :, 0, :, 1] +
              x[:, :, :, 1, :, 0] + x[:, :, :, 1, :, 1]) / 4.0
        HH = (x[:, :, :, 0, :, 0] - x[:, :, :, 0, :, 1] -
              x[:, :, :, 1, :, 0] + x[:, :, :, 1, :, 1]) / 4.0

        # 返回独立子带，而非拼接（让二次分解能正确处理）
        return LL, LH, HL, HH


class IWT(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        C = C // 4  # 假设输入是4c通道（拼接后的）
        LL, LH, HL, HH = x[:, 0:C], x[:, C:2 * C], x[:, 2 * C:3 * C], x[:, 3 * C:4 * C]

        out = torch.zeros(B, C, H * 2, W * 2).to(x.device)
        out[:, :, 0::2, 0::2] = (LL - LH - HL + HH) / 2.0
        out[:, :, 0::2, 1::2] = (LL + LH - HL - HH) / 2.0
        out[:, :, 1::2, 0::2] = (LL - LH + HL - HH) / 2.0
        out[:, :, 1::2, 1::2] = (LL + LH + HL + HH) / 2.0
        return out

# --------------------- BRE Block ---------------------

class BRE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dwt = DWT()
        self.dwt_ll = DWT()
        self.iwt = IWT()

        c = in_channels
        self.conv_fl_ll = nn.Conv2d(c, c, 3, padding=1)
        self.conv_fl_hf = nn.Conv2d(3*c, c, 1)  # 高频子带是3个，通道数改为3c
        self.pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.ll_fuse = nn.Conv2d(2 * c, c, 1)
        self.hf_conv1 = nn.Conv2d(3 * c, c, 3, padding=1)  # 一次分解的高频是3个，通道数3c
        self.hf_conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.final_conv = nn.Conv2d(2 * c, c, 3, padding=1)

    def forward(self, x, is_source=None):
        B, C, H, W = x.shape
        res = x

        # 第一次DWT：获取独立子带
        LL, LH, HL, HH = self.dwt(x)  # 1低频 + 3高频

        # 第二次DWT：对LL执行二次分解，获取独立子带
        LL_ll, LL_lh, LL_hl, LL_hh = self.dwt_ll(LL)  # 1低频(FLF) + 3高频(FHF)

        # 低频分支：FLF是二次分解的低频子带
        FLF = LL_ll
        # 图示：3x3 Conv + ReLU
        FLF = self.relu(self.conv_fl_ll(FLF))  # 加上 ReLU

        # 高频分支（二次分解的高频）：3个分量拼接为FHF
        FHF = torch.cat([LL_lh, LL_hl, LL_hh], dim=1)  # 通道数3c
        # 图示：MaxPool → 1x1 Conv → ReLU → 调整顺序
        FHF = self.pool(FHF)  # 先 MaxPool
        FHF = self.conv_fl_hf(FHF)  # 再 1x1 Conv
        FHF = self.relu(FHF)  # 最后 ReLU

        # 上采样保持不变（为了和 FLF 尺寸对齐）
        FHF = F.interpolate(FHF, size=FLF.shape[2:], mode='bilinear', align_corners=False)  # 上采样

        # 低频分支融合：FLF + FHF（残差连接）
        FL_ = self.ll_fuse(torch.cat([FLF, FHF], dim=1))  # 通道拼接
        FL_ = F.interpolate(FL_, size=LL.shape[2:], mode='bilinear', align_corners=False)
        FL_ = FL_ + LL  # 与第一次分解的LL残差连接（贴合架构图）

        # 高频分支（一次分解的高频）：3个分量拼接为FH
        FH = torch.cat([LH, HL, HH], dim=1)  # 通道数3c
        FH_ = self.hf_conv2(self.hf_conv1(FH))  # 两次卷积

        # 空间对齐
        min_h = min(FL_.shape[2], FH_.shape[2])
        min_w = min(FL_.shape[3], FH_.shape[3])
        FL_ = FL_[:, :, :min_h, :min_w]
        FH_ = FH_[:, :, :min_h, :min_w]

        # 最终融合：FL_ + FH_
        merged = self.final_conv(torch.cat([FL_, FH_], dim=1))  # 通道拼接

        # IWT重构：需要4个子带拼接（同步修改IWT类）
        out = self.iwt(torch.cat([merged] * 4, dim=1))

        # 残差连接（与输入x对齐）
        if out.shape != res.shape:
            res = F.interpolate(res, size=out.shape[2:], mode='bilinear', align_corners=False)
        return out + res
# --------------------- Backbone & Head Integration ---------------------

class Backbone(nn.Module):
    def __init__(self, resnet, use_filter):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        if use_filter:
            self.layer1 = add_module_after_block(resnet.layer1, 1, BRE(256))
            self.layer2 = add_module_after_block(resnet.layer2, 1, BRE(512))
            self.layer3 = add_module_after_block(resnet.layer3, 1, BRE(1024))
        else:
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
        self.out_channels = 1024

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if torch.isnan(x).int().sum() > 0:
            print(torch.isnan(x).int().sum())
        return x

    def forward(self, x):
        feat = self._forward(x)
        return OrderedDict([['feat_res4', feat]])


class Res5Head(nn.Module):
    def __init__(self, layer4, use_filter):
        super().__init__()
        self.layer4 = deepcopy(layer4)
        if use_filter:
            self.layer4 = add_module_after_block(self.layer4, 1, BRE(2048))
        self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = self.layer4(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([['feat_res4', x], ['feat_res5', feat]])


class ReidRes5Head(nn.Module):
    def __init__(self, layer4, use_filter):
        super().__init__()
        self.layer4 = deepcopy(layer4)
        if use_filter:
            self.layer4 = add_module_after_block(self.layer4, 1, BRE(2048))
        self.out_channels = [1024, 2048]

    def bottleneck_forward(self, bottleneck, x, is_source):
        identity = x
        out = bottleneck.conv1(x)
        out = bottleneck.bn1(out, is_source) if isinstance(bottleneck.bn1, DSBN2d) else bottleneck.bn1(out)
        out = bottleneck.relu(out)
        out = bottleneck.conv2(out)
        out = bottleneck.bn2(out, is_source) if isinstance(bottleneck.bn2, DSBN2d) else bottleneck.bn2(out)
        out = bottleneck.relu(out)
        out = bottleneck.conv3(out)
        out = bottleneck.bn3(out, is_source) if isinstance(bottleneck.bn3, DSBN2d) else bottleneck.bn3(out)
        if bottleneck.downsample is not None:
            for module in bottleneck.downsample:
                identity = module(identity, is_source) if isinstance(module, DSBN2d) else module(identity)
        out += identity
        out = bottleneck.relu(out)
        return out

    def forward(self, x, is_source=True):
        module_seq = []
        for _, (_, child) in enumerate(self.named_modules()):
            if isinstance(child, torchvision.models.resnet.Bottleneck):
                module_seq.append(child)
            if isinstance(child, BRE):
                module_seq.append(child)
        feat = x.clone()
        for module in module_seq:
            feat = module(feat, is_source) if isinstance(module, BRE) else self.bottleneck_forward(module, feat, is_source)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([['feat_res4', x], ['feat_res5', feat]])


# --------------------- Build Entry ---------------------

def build_resnet(name='resnet50', pretrained=True):
    resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)
    return (
        Backbone(resnet, use_filter=True),
        Res5Head(resnet.layer4, use_filter=False),
        ReidRes5Head(resnet.layer4, use_filter=True),
    )