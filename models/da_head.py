# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from models.da_loss import DALossComputation

# Faster R-CNN 结构中的跨域目标检测模块，其核心目标是：通过图像级 + 实例级对抗训练，使模型具备从源域迁移到目标域的能力



#这是对 GRL (Gradient Reversal Layer) 的实现，本质功能：前向：直接输出 input     反向：将反向传播的梯度乘以一个 负值权重（如 -0.1）相当于：“鼓励主干特征对抗域分类器，使其无法区分源域与目标域”
class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight * grad_input, None


gradient_scalar = _GradientScalarLayer.apply

#对抗训练 通过 GRL，使特征生成器学到“域不可辨别”的表征
class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "weight=" + str(self.weight)
        tmpstr += ")"
        return tmpstr

#图像级判别 判断整张图像是否来自源/目标域，用于监督全局对抗
class DAImgHead(nn.Module):    #输出：大小为 (B, 1, H, W) 的“图像级域判别图”，每个位置一个“是否为目标域”的分数
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super().__init__()

        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        img_features = []
        for feature in x:
            t = F.relu(self.conv1_da(feature))
            img_features.append(self.conv2_da(t))
        return img_features

#实例级判别 判断每个 proposal 是否来自源/目标域
class DAInsHead(nn.Module): #fc1 → ReLU → Dropout → fc2 → ReLU → Dropout → fc3 → Sigmoid
    """
    Adds a simple Instance-level Domain Classifier head
    """
    # 输入：每个 proposal 的 pooled feature vector
    # 输出：每个proposal属于目标域的概率（域标签）
    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x


class DomainAdaptationModule(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self, DA_HEADS):
        super().__init__()

        # self.cfg = cfg.clone()

        stage_index = 4   #这个用于计算实例级特征的维度（backbone 第4层输出）。
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = 256  # cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_ins_inputs = res2_out_channels * stage2_relative_factor

        # self.resnet_backbone = cfg.MODEL.BACKBONE.CONV_BODY.startswith('R')
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

        self.img_weight = DA_HEADS.DA_IMG_LOSS_WEIGHT  #损失函数的权重从 DA_HEADS 里读取
        self.ins_weight = DA_HEADS.DA_INS_LOSS_WEIGHT   #分别控制图像级、实例级、一致性损失的比例。
        self.cst_weight = DA_HEADS.DA_CST_LOSS_WEIGHT

        self.grl_img = GradientScalarLayer(-1.0 * 0.1)  #定义多个梯度反转层 GRL  通过梯度反转来鼓励模型学习域不变的特征
        self.grl_ins = GradientScalarLayer(-1.0 * 0.1)  #正负号控制“主损失”和“辅助一致性约束”
        self.grl_ins_before = GradientScalarLayer(-1.0 * 0.1)
        self.grl_img_consist = GradientScalarLayer(1.0 * 0.1)
        self.grl_ins_consist = GradientScalarLayer(1.0 * 0.1)
        self.grl_ins_consist_before = GradientScalarLayer(1.0 * 0.1)

        in_channels = 256 * 4  # cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.lw_da_ins = DA_HEADS.LW_DA_INS

        self.imghead = DAImgHead(1024)  #对图像特征图进行域分类（使用 conv）
        self.inshead = DAInsHead(256)  #对实例特征进行域分类（使用 fc）
        self.inshead_before = DAInsHead(2048)  #对更早期的实例特征进行判别（前层信息）
        self.loss_evaluator = DALossComputation()  #用于计算上面几种损失函数  引用的是 da_head.py里面的函数  来计算 领域自适应损失

    def forward(
        self, img_features, da_ins_feature, da_ins_labels, da_ins_feature_before, da_ins_labels_before, targets=None
    ):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are                    img_features: 来自 backbone 的多层图像特征
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors                                    da_ins_feature: 实例级特征（RoI pooled）
            da_ins_labels (Tensor): domain labels for instance-level feature vectors                   da_ins_labels: 域标签（如 source = 1, target = 0）
            targets (list[BoxList): ground-truth boxes present in the image (optional)                 da_ins_feature_before: 较浅层提取的实例特征
                                                                                                       targets: 真实 box，仅用于一致性损失时辅助训练
        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        #reshape 实例特征为向量
        da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)
        da_ins_feature_before = da_ins_feature_before.view(da_ins_feature_before.size(0), -1)
        #对抗训练的特征生成（GRL 处理） 对图像特征、实例特征施加梯度反转，准备送入 domain classifier
        img_grl_fea = [self.grl_img(fea) for fea in img_features]
        ins_grl_fea = self.grl_ins(da_ins_feature)
        ins_grl_fea_before = self.grl_ins_before(da_ins_feature_before)
        #一致性训练路径的 GRL 处理   一致性分支用于使 image-level 与 instance-level 的判断保持一致。
        img_grl_consist_fea = [self.grl_img_consist(fea) for fea in img_features]
        ins_grl_consist_fea = self.grl_ins_consist(da_ins_feature)
        ins_grl_consist_fea_before = self.grl_ins_consist_before(da_ins_feature_before)
        #分别过 Head 得到预测结果
        da_img_features = self.imghead(img_grl_fea)
        da_ins_features = self.inshead(ins_grl_fea)
        da_ins_features_before = self.inshead_before(ins_grl_fea_before)

        da_img_consist_features = self.imghead(img_grl_consist_fea)   #分别得到两个路径（原始判别与一致性约束）的输出
        da_ins_consist_features = self.inshead(ins_grl_consist_fea)
        da_ins_consist_features_before = self.inshead_before(ins_grl_consist_fea_before)
        #这些被用于一致性损失，sigmoid 处理后可以解释为概率
        da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]
        da_ins_consist_features = da_ins_consist_features.sigmoid()
        da_ins_consist_features_before = da_ins_consist_features_before.sigmoid()
        if self.training:  # 损失计算（仅在训练时）  第一次损失计算（当前特征）
            da_img_loss, da_ins_loss, da_consistency_loss = self.loss_evaluator(
                da_img_features,
                da_ins_features,
                da_img_consist_features,
                da_ins_consist_features,
                da_ins_labels,
                targets,
            )
            da_img_loss, da_ins_loss_before, da_consistency_loss_before = self.loss_evaluator(  #第二次损失计算（历史特征）
                da_img_features,
                da_ins_features_before,
                da_img_consist_features,
                da_ins_consist_features_before,
                da_ins_labels_before,
                targets,
            )
            losses = {}
            if self.img_weight >= 0:
                losses["loss_da_image"] = self.img_weight * da_img_loss
            if self.ins_weight >= 0:
                losses["loss_da_instance"] = self.ins_weight * (
                    self.lw_da_ins * da_ins_loss + (1.0 - self.lw_da_ins) * da_ins_loss_before
                )
                # losses["loss_da_instance"] = self.ins_weight * da_ins_loss_before
            if self.cst_weight >= 0:
                losses["loss_da_consistency"] = self.cst_weight * (
                    self.lw_da_ins * da_consistency_loss + (1.0 - self.lw_da_ins) * da_consistency_loss_before   #使用 lw_da_ins 控制当前与历史损失的加权
                )
                # losses["loss_da_consistency"] = self.cst_weight * da_consistency_loss_before
            return losses  #最终返回包含图像级、实例级、一致性三种损失
        return {}

# 该模块实现了完整的 多粒度、多路径、可控损失组合 的 域对抗模块，关键技术包括：
# 	•	梯度反转层（GRL）用于反向对抗训练
# 	•	图像级 vs. 实例级 对抗
# 	•	当前特征 vs. 历史特征 双路训练
# 	•	一致性正则化（Consistency Loss）增强鲁棒性