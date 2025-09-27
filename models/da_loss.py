import torch
from torch import nn
from torch.nn import functional as F


def consistency_loss(img_feas, ins_fea, ins_labels, size_average=True):  #目的是让每个实例级的 domain 判断值接近其所在图像的 domain 判断值的平均值
    """
    Consistency regularization as stated in the paper
    `Domain Adaptive Faster R-CNN for Object Detection in the Wild`
    L_cst = \\sum_{i,j}||\frac{1}{|I|}\\sum_{u,v}p_i^{(u,v)}-p_{i,j}||_2
    """
    loss = []
    len_ins = ins_fea.size(0)
    # intervals = [torch.nonzero(ins_labels).size(0), len_ins-torch.nonzero(ins_labels).size(0)]
    for img_fea_per_level in img_feas:
        N, A, H, W = img_fea_per_level.shape
        img_fea_per_level = torch.mean(img_fea_per_level.reshape(N, -1), 1)
        img_feas_per_level = []
        # assert N==2, \
        #     "only batch size=2 is supported for consistency loss now, received batch size: {}".format(N)
        for i in range(N):
            # img_fea_mean = img_fea_per_level[i].view(1, 1).repeat(intervals[i], 1)
            img_fea_mean = img_fea_per_level[i].view(1, 1).repeat(len_ins // N, 1)
            img_feas_per_level.append(img_fea_mean)
        if len_ins % N != 0:
            img_feas_per_level.append(img_fea_per_level[N - 1].view(1, 1).repeat(len_ins % N, 1))
        img_feas_per_level = torch.cat(img_feas_per_level, dim=0)  #计算 L1 差异（可理解为 domain 分布差异）
        loss_per_level = torch.abs(img_feas_per_level - ins_fea)
        loss.append(loss_per_level)
    loss = torch.cat(loss, dim=1)
    if size_average:
        return loss.mean()
    return loss.sum()


class DALossComputation:  #DALossComputation 类分析（领域对抗损失模块）
    """
    This class computes the DA loss.
    """

    def __init__(self):
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

    def prepare_masks(self, targets):  #根据 target["domain_labels"] 来判断这张图是不是 source 域：
        masks = []
        for targets_per_image in targets:
            is_source = targets_per_image["domain_labels"]
            mask_per_image = (   #最终将每张图的 source/target 标记组成 mask 列表
                is_source.new_ones(1, dtype=torch.bool) if is_source.any() else is_source.new_zeros(1, dtype=torch.bool)
            )
            masks.append(mask_per_image)
        return masks

    def __call__(self, da_img, da_ins, da_img_consist, da_ins_consist, da_ins_labels, targets):
        """
        Arguments:
            da_img (list[Tensor])
            da_img_consist (list[Tensor])
            da_ins (Tensor)
            da_ins_consist (Tensor)
            da_ins_labels (Tensor)
            targets (list[BoxList])

        Returns:
            da_img_loss (Tensor)
            da_ins_loss (Tensor)
            da_consist_loss (Tensor)
        """

        masks = self.prepare_masks(targets) #获取图像级 mask
        masks = torch.cat(masks, dim=0)

        da_img_flattened = []    #图像级特征处理
        da_img_labels_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the image-level domain alignment
        for da_img_per_level in da_img:   #遍历每个图像层级的 domain logit
            N, A, H, W = da_img_per_level.shape
            da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)  #permute 为 (N, H, W, C)，以匹配真实标签
            da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32) #构造标签：如果是 source 域，对应位置设为 1
            da_img_label_per_level[masks, :] = 1

            da_img_per_level = da_img_per_level.reshape(N, -1)
            da_img_label_per_level = da_img_label_per_level.reshape(N, -1)  #flatten 成为 2D 向量（[N, H*W]）

            da_img_flattened.append(da_img_per_level)
            da_img_labels_flattened.append(da_img_label_per_level)

        da_img_flattened = torch.cat(da_img_flattened, dim=0) #拼接所有层级并计算图像级损失
        da_img_labels_flattened = torch.cat(da_img_labels_flattened, dim=0)

        da_img_loss = F.binary_cross_entropy_with_logits(da_img_flattened, da_img_labels_flattened)  #实例级损失：与 domain 标签进行 BCE 损失
        da_ins_loss = F.binary_cross_entropy_with_logits(
            torch.squeeze(da_ins), da_ins_labels.type(torch.cuda.FloatTensor)
        )

        da_consist_loss = consistency_loss(da_img_consist, da_ins_consist, da_ins_labels, size_average=True)  #一致性正则项（图像平均表示和实例保持接近）
        # da_consist_loss = 0

        return da_img_loss, da_ins_loss, da_consist_loss  #返回所有损失


# 图像级判别损失   da_img_loss   监督整张图像来自 source 或 target
# 实例级判别损失   da_ins_loss   监督目标框的实例来自 source 或 target
# 一致性损失       da_consist_loss   保证 image-level 与 instance-level 的 domain 判断保持一致
# 这个类最终被 DomainAdaptationModule 调用，并根据权重系数整合为总损失，进而用于领域对抗训练。