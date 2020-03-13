# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self, 
        proposal_matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    # 计算出所有预测边框所对应的基准边框(groun truth box),并返回对应的列表
    def match_targets_to_proposals(self, proposal, target, is_source=True):
        """
        这个函数主要是计算一下proposal以及gt的IOU，然后筛选一下
        相当于是选择出来了每个proposal对应的gt
        """
        # 计算基准边框与预测边框相互之间的IoU
        match_quality_matrix = boxlist_iou(target, proposal)
        # 计算各个预测边框对应的基准边框(ground truth box)的索引列表，背景边框为-2，模糊边框为-1
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        # 获得基准边框(groun truth box)附加的属性labels标签,即边框的具体类别
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        # 计算各个预测边框对应的基准边框(ground truth box)列表,所有背景边框以及模糊边框都对应成第一个gt
        matched_targets = target[matched_idxs.clamp(min=0)]

        # DA start
        # 如果是目标域的数据的话
        if not is_source:
            matched_targets = target[matched_idxs]
        # DA end
        # 将对应的列表索引附加到对应基准边框列表中
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets, sample_for_da=False):
        """"
        计算出所有预测边框所对应的基准边框(groun truth box)
        """
        # 在这个函数中生成的东西就存在这里
        labels = []
        regression_targets = []
        domain_labels = []
        # 针对每一张图片计算预测边框对应的基准边框列表
        # python的zip函数可以将两个list组合起来，形成tuple组
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            # DA start
            is_source = targets_per_image.get_field('is_source')
            # 得到各个预测边框对应的基准边框(ground truth box)列表,所有背景边框以及模糊边框都对应成第一个gt
            # 调用的这个match_targets_to_proposals函数也变了
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image, is_source.any()
            )
            # DA end
            # 获得对应的列表索引
            matched_idxs = matched_targets.get_field("matched_idxs")
            # 获得每一张图片生成的预测边框对应的具体类别标签，并将其转换为相应的数据类型
            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            # 获得背景边框列表的索引
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            # 把背景的预测边框对应的边框类别设置为0
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            # 获得模糊边框列表的索引
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            # 把模糊的预测边框对应的边框类别设置为-1
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            # 计算proposal对应的gt框和proposal框之间的偏差值
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )
            # DA start
            # 怎么根据这里的代码，S的label是1，T的label是0，但是根据论文，这是相反的啊，在后面--call--函数有解释
            domain_label = torch.ones_like(labels_per_image, dtype=torch.uint8) if is_source.any() \
                else torch.zeros_like(labels_per_image, dtype=torch.uint8)
            domain_labels.append(domain_label)
            # 如果是目标域的数据
            if not is_source.any():
                labels_per_image[:] = 0
            # 如果是针对DA的采样
            if sample_for_da:
                labels_per_image[:] = 0
            # DA end
            # 将预测边框列表对应边框类别添加到标签列表中
            labels.append(labels_per_image)
            # 添加边框回归列表
            regression_targets.append(regression_targets_per_image)
        # DA start
        return labels, regression_targets, domain_labels
        # DA end

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])

        下面我对这个函数做一个总结：
        1.通过prepare_targets函数
            计算proposal与gt之间的IOU，找出来与每个proposal对应的gt的labels，对label置0,1,-1
            计算proposal与gt之间的变换关系regression
            计算proposal的domain label
        2.根据label筛选出来pos和neg的proposal，同时对proposal添加label，regression以及domain label信息
        """
        # DA start
        # 得到预测边框的类别标签以及边框回归信息列表
        labels, regression_targets, domain_labels = self.prepare_targets(proposals, targets)
        # DA end
        # 按照一定方式选取背景边框和目标边框,并返回其标签,在label中1为目标,0为背景
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        # 按照图片,将额外属性保存到边框列表中
        for labels_per_image, regression_targets_per_image, proposals_per_image, domain_label_per_image \
                in zip(labels, regression_targets, proposals, domain_labels):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )
            # DA start
            proposals_per_image.add_field("domain_labels", domain_label_per_image)
            # DA end

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        # 从每一张图片中提取背景边框和目标边框
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    # DA start 这里整个函数都是
    def subsample_for_da(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])

        参考上面的subsample里面的介绍
        """
        # 注意，这里的最后一个参数是True，得到的labels都是0
        labels, _, domain_labels = self.prepare_targets(proposals, targets, sample_for_da=True)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and information to the bounding boxes
        for proposals_per_image, domain_label_per_image in zip(
            proposals, domain_labels
        ):
            proposals_per_image.add_field("domain_labels", domain_label_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image
        # 这里少了那句self._proposals = proposals
        return proposals
    # DA end

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.
        计算Faster R-CNN的损失值，这一操作需要subsample方法已经在之前被调用过
        参数:
            class_logits (list[Tensor])：类别信息数据
            box_regression (list[Tensor])：边框信息回归数据
        返回值:
            classification_loss (Tensor)：分类误差损失值
            box_loss (Tensor)：边框回归偏差损失值
        """
        # 将类别和边框信息矩阵连接起来，并去掉无用的维度
        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        # 获得设备名称
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        # 注意这里得到的是sample那个函数那时候的那个proposal
        proposals = self._proposals
        # 将不同图片的预测边框的标签连接起来
        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        # 将不同图片的预测边框的回归值列表合并起来
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        # DA start
        # 将不同图片的domain-label信息统一起来
        # 需要注意的是，这个domain-mask源域是1，目标域是0，这正好对应了下面四句代码提取的时候提取到source的
        domain_masks = cat([proposal.get_field("domain_labels") for proposal in proposals], dim=0)

        class_logits = class_logits[domain_masks, :]
        box_regression = box_regression[domain_masks, :]
        labels = labels[domain_masks]
        regression_targets = regression_targets[domain_masks, :]

        # DA end
        # 计算所有预测边框类别信息的损失值
        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        # 获得预测边框标签中>0的索引，即获得有目标的预测边框在边框列表中的索引
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        # 获得所有含有目标的预测边框的标签列表
        labels_pos = labels[sampled_pos_inds_subset]
        # 如果整个模型采用agnostic模型，即只分别含目标与不含目标两类
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            # 当时正常模式时，获得含有目标的边框在对应的边框回归信息矩阵中的索引
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)
        # 计算边框回归信息的损失值
        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()
        # DA start
        # 返回的这个domain-masks里面，源域是1，目标域是0
        return classification_loss, box_loss, domain_masks
        # DA end

def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
