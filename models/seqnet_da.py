import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

from models.resnet import build_resnet  #构建resnet骨干网络
from models.roi_head_da import SeqRoIHeadsDa
from models.da_head import DomainAdaptationModule
from models.box_head import BBoxRegressor
from apex import amp


class SeqNetDa(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.target_start_epoch = cfg.TARGET_REID_START

        backbone, box_head, reid_head = build_resnet(name="resnet50", pretrained=True)

        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))  #定义每个尺度上使用的 anchor 尺寸与比例。
        head = RPNHead(  #RPN 头部结构：分类 + 回归
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(training=cfg.MODEL.RPN.PRE_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.PRE_NMS_TOPN_TEST)  #控制训练和测试中保留 proposal 数量的参数
        post_nms_top_n = dict(training=cfg.MODEL.RPN.POST_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.POST_NMS_TOPN_TEST)
        rpn = RegionProposalNetwork(    #实例化 RPN，控制 proposal 的筛选逻辑与正负样本划分
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        )
        #ROI Heads 初始化
        faster_rcnn_predictor = FastRCNNPredictor(2048, 2)  #faster_rcnn_predictor: 用于分类任务
        # reid_head = deepcopy(box_head)
        box_roi_pool = MultiScaleRoIAlign(featmap_names=["feat_res4"], output_size=14, sampling_ratio=2) #box_roi_pool: 将不同尺寸的 proposal 对齐到固定特征尺寸
        box_predictor = BBoxRegressor(2048, num_classes=2, bn_neck=cfg.MODEL.ROI_HEAD.BN_NECK)  #box_predictor: 用于回归
        roi_heads = SeqRoIHeadsDa(     #SeqRoIHeadsDa: 自定义 RoI Head，加入 ReID 和域判别逻辑
            # SeqNet
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            # parent class
            box_roi_pool=box_roi_pool, #为 MultiScaleRoIAlign函数
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN, # 被认为是前景的RoI重叠阈值(如果>= POS_THRESH_TRAIN) 0.5
            bg_iou_thresh=cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN, #背景区域的重叠阈值(如果< NEG_THRESH_TRAIN) 0.5
            batch_size_per_image=cfg.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN, #用于训练RoI头部的每幅图像的RoI个数 128
            positive_fraction=cfg.MODEL.ROI_HEAD.POS_FRAC_TRAIN, #每个RoI小批量前景示例的目标比例 0.5
            bbox_reg_weights=None,
            score_thresh=cfg.MODEL.ROI_HEAD.SCORE_THRESH_TEST, #最低分数阈值 0.5
            nms_thresh=cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST,  #NMS threshold used on boxes 0.4
            detections_per_img=cfg.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,  #最大检测到的对象数 300
        )

        transform = GeneralizedRCNNTransform(  #图像归一化与尺度缩放处理，兼容 torchvision API
            min_size=cfg.INPUT.MIN_SIZE,
            max_size=cfg.INPUT.MAX_SIZE,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
# here modified, for adapting to amp  这里需要注释掉 不然会报input和roi不匹配的精度
        #self.roi_heads.box_roi_pool.forward = amp.half_function(self.roi_heads.box_roi_pool.forward)
        #作者原始代码跑的话，这段需要注释掉才行，现在我自己修改的代码 方法2，需要这段话！
        #self.roi_heads.box_roi_pool.forward = amp.half_function(self.roi_heads.box_roi_pool.forward)
        self.transform = transform
        self.da_heads = DomainAdaptationModule(cfg.MODEL.DA_HEADS)    #加入自定义的 DomainAdaptationModule，包含图像级、实例级和一致性损失

        # loss weights   将各类损失（检测、回归、ReID、DA）对应的权重配置记录下来，用于后续加权
        self.lw_rpn_reg = cfg.SOLVER.LW_RPN_REG
        self.lw_rpn_cls = cfg.SOLVER.LW_RPN_CLS
        self.lw_proposal_reg = cfg.SOLVER.LW_PROPOSAL_REG
        self.lw_proposal_cls = cfg.SOLVER.LW_PROPOSAL_CLS
        self.lw_box_reg = cfg.SOLVER.LW_BOX_REG
        self.lw_box_cls = cfg.SOLVER.LW_BOX_CLS
        self.lw_box_reid = cfg.SOLVER.LW_BOX_REID
        self.lw_box_reid_t = cfg.SOLVER.LW_BOX_REID_T

    # The is_source here should be switched when inferencing  推理阶段：inference 函数
    def inference(self, images, targets=None, query_img_as_gallery=False, is_source=False): #推理函数入口，接收图像、目标（可选）、是否把查询图像当作 gallery、当前是否是源域（source）等参数。
        original_image_sizes = [img.shape[-2:] for img in images]  #这几行首先处理输入的图像数据。images 经过了大小的变换，并用骨干网络 backbone 得到了特征。 记录每张图像的原始尺寸，用于后续结果恢复
        images, targets = self.transform(images, targets) #对图像进行标准化、尺寸调整，统一输入格式
        features = self.backbone(images.tensors) #用主干网络提取图像特征

        if query_img_as_gallery: #在查询图像作为图库时，需要确认是否提供了目标信息。
            assert targets is not None

        # 如果存在目标信息且不是查询图像作为图库，模型会处理检测目标。它会从目标中提取边界框信息 boxes，
        # 然后在 RoI pooling 层中将这些信息转换成特征向量，最后将这些特征向量输入到嵌入头 embedding_head 中。
        if targets is not None and not query_img_as_gallery:
            # query
            boxes = [t["boxes"] for t in targets]

            box_features = self.roi_heads.box_roi_pool(features, boxes, images.image_sizes)
            box_features = self.roi_heads.reid_head(box_features, is_source)
            embeddings, _ = self.roi_heads.embedding_head(box_features)
            return embeddings.split(1, 0)
        # 否则，会生成提议框，然后通过 RoIHeads 模块处理这些提议框，最后进行后处理步骤，将得到的检测结果进行转换后返回。
        else:
            # gallery  生成anchors -> softmax分类器提取positvie anchors -> bbox reg回归positive anchors -> Proposal Layer生成proposals
            # 而RoI Pooling层则负责收集proposal，并计算出proposal feature maps，送入后续网络。
            proposals, _ = self.rpn(images, features, targets)
            detections, _ = self.roi_heads(
                features, proposals, images.image_sizes, targets, query_img_as_gallery, is_source
            )
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            return detections

    def forward(
        self,
        images_s,
        targets_s=None,
        images_t=None,
        targets_t=None,
        query_img_as_gallery=False,
        is_source=False,
        epoch=0,
    ):
        if not self.training:
            return self.inference(images_s, targets_s, query_img_as_gallery, is_source)   #不是训练到话 就是inference  这里加了is_source 的判断

        images_s, targets_s = self.transform(images_s, targets_s)  #处理源域、目标域图像。
        images_t, targets_t = self.transform(images_t, targets_t)

        losses = {}
        features_s = self.backbone(images_s.tensors)  # images_s 源域
        proposals_s, proposal_losses_s = self.rpn(images_s, features_s, targets_s)  #源域图像通过主干网络提特征，再进入 RPN 得到 proposals 和 RPN 损失
        _, detector_losses_s = self.roi_heads(features_s, proposals_s, images_s.image_sizes, targets_s)  #RoIHeads 进一步提取 proposal 特征，做分类、框回归和 ReID
##这些是不一样的地方！  从 RoI 中提取 instance-level 的特征，构造源域的对抗标签
        da_ins_feas_s, da_ins_labels_s, da_ins_feas_s_before, da_ins_labels_s_before = self.roi_heads.extract_da(
            features_s, proposals_s, images_s.image_sizes, targets_s
        )
        da_ins_labels_s = torch.cat(da_ins_labels_s)
        da_ins_labels_s_before = torch.cat(da_ins_labels_s_before)

        # rename rpn losses to be consistent with detection losses RPN 损失命名统一化（便于后续加权）
        proposal_losses_s["loss_rpn_reg"] = proposal_losses_s.pop("loss_rpn_box_reg") #相当于更名一下 回归损失
        proposal_losses_s["loss_rpn_cls"] = proposal_losses_s.pop("loss_objectness")  #分类损失
#这些是不一样的地方！ 目标域同样进行提特征和 RPN
        features_t = self.backbone(images_t.tensors)
        proposals_t, proposal_losses_t = self.rpn(images_t, features_t, targets_t)

        if epoch >= self.target_start_epoch: #如果到达设定 epoch，开始对目标域执行 ReID
            _, reid_losses_t = self.roi_heads(
                features_t, proposals_t, images_t.image_sizes, targets_t, query_img_as_gallery=False, is_source=False
            )

            # rename target domain losses 命名为 _t 表示 target domain；加入总损失字典
            proposal_losses_t["loss_rpn_reg_t"] = proposal_losses_t.pop("loss_rpn_box_reg")
            proposal_losses_t["loss_rpn_cls_t"] = proposal_losses_t.pop("loss_objectness")
            reid_losses_t["loss_box_reg_t"] = reid_losses_t.pop("loss_box_reg")
            reid_losses_t["loss_box_cls_t"] = reid_losses_t.pop("loss_box_cls")
            reid_losses_t["loss_proposal_reg_t"] = reid_losses_t.pop("loss_proposal_reg")
            reid_losses_t["loss_proposal_cls_t"] = reid_losses_t.pop("loss_proposal_cls")
            losses.update(reid_losses_t)
            losses.update(proposal_losses_t)
#目标域（TargetDomain）是无标签或弱监督状态，因此其损失不应主导整体训练，否则容易干扰源域（SourceDomain）的稳定优化过程。•特别是前面几项乘以 0.1 * self.lw_xxx，保留了一定贡献，但降低影响力。
            losses["loss_rpn_reg_t"] *= 0.1 * self.lw_rpn_reg  #目标域的损失通常乘上较小权重（如 0.1），防止干扰源域收敛
            losses["loss_rpn_cls_t"] *= 0.1 * self.lw_rpn_cls
            losses["loss_proposal_reg_t"] *= 0.1 * self.lw_proposal_reg
            losses["loss_proposal_cls_t"] *= 0.1 * self.lw_proposal_cls
            losses["loss_box_reg_t"] *= 0.1 * self.lw_box_reg
            losses["loss_box_cls_t"] *= 0.1 * self.lw_box_cls
            losses["loss_box_reid_t"] *= self.lw_box_reid_t
#目标域 DA 特征提取 + DA Loss 计算    提取目标域 DA 特征，准备输入 DA Head
        da_ins_feas_t, da_ins_labels_t, da_ins_feas_t_before, da_ins_labels_t_before = self.roi_heads.extract_da(
            features_t, proposals_t, images_t.image_sizes, targets_t
        )
        da_ins_labels_t = torch.cat(da_ins_labels_t)
        da_ins_labels_t_before = torch.cat(da_ins_labels_t_before)
        if self.da_heads:  #分别计算源域、目标域的   对抗损失（图像级 + 实例级 + 一致性损失）
            da_losses_s = self.da_heads(
                [features_s["feat_res4"]],
                da_ins_feas_s,
                da_ins_labels_s,
                da_ins_feas_s_before,
                da_ins_labels_s_before,
                targets_s,
            )
            da_losses_t = self.da_heads(
                [features_t["feat_res4"]],
                da_ins_feas_t,
                da_ins_labels_t,
                da_ins_feas_t_before,
                da_ins_labels_t_before,
                targets_t,
            )

        losses.update(detector_losses_s)
        losses.update(proposal_losses_s)
        da_losses_t["loss_da_image_t"] = da_losses_t.pop("loss_da_image") #对目标域的 DA 损失重命名（加 _t）
        da_losses_t["loss_da_instance_t"] = da_losses_t.pop("loss_da_instance")
        da_losses_t["loss_da_consistency_t"] = da_losses_t.pop("loss_da_consistency")
        losses.update(da_losses_s)  #汇总所有损失项
        losses.update(da_losses_t)
# 对应源域的损失项做了加权（没有显式缩小目标域DA部分的损失权重
        # apply loss weights 应用损失权重（仅对源域） 对源域 RPN、Proposal、RoI Head 的各项损失乘以设置的超参数
        losses["loss_rpn_reg"] *= self.lw_rpn_reg
        losses["loss_rpn_cls"] *= self.lw_rpn_cls
        losses["loss_proposal_reg"] *= self.lw_proposal_reg
        losses["loss_proposal_cls"] *= self.lw_proposal_cls
        losses["loss_box_reg"] *= self.lw_box_reg
        losses["loss_box_cls"] *= self.lw_box_cls
        losses["loss_box_reid_s"] *= self.lw_box_reid

        return losses
