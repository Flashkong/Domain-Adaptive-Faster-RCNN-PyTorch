# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.

    在FasterRCNN或者MaskRCNN中，在通过RPN筛选出大小不同的一系列预测边框后，需要在Box_Head层对选出的预测边框进行进一步筛选，
    去除掉边框IoU介于两种阈值之间的预测边框，然后再使用ROI Pooling将处理后的预测边框池化为大小一致的边框。
    然后对这些边框进行进一步的特征提取，之后再在提取后的特征上进行进一步的边框预测，
    得到每一个预测边框的类别得分以及它们的边框回归信息，并以此来得到最终的预测边框以及其类别信息。
    原文链接：https://blog.csdn.net/leijiezhang/article/details/92063984
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        # 指定ROI层中box_head模块的特征提取类
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        # 指定ROI层中box_head模块的边框预测类
        # 本项目中，使用的predictor是FastRCNNPredictor而不是FPNPredictor
        self.predictor = make_roi_box_predictor(cfg)
        # 指定box_head模块特征提取后的一系列操作，包括修正预测边框等
        self.post_processor = make_roi_box_post_processor(cfg)
        # 指定box_head模块的loss评估类
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels。从backbone中获取的特征图（可能是多层）
            proposals (list[BoxList]): proposal boxes 预测边框
            targets (list[BoxList], optional): the ground-truth targets.基准边框

        Returns:
            x (Tensor): the result of the feature extractor 在重新修正后的预测边框中提取的特征信息
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned 重新修正后的最终的预测边框
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict. 本层的loss值
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            # 在上层得出的预测边框里面，经过特征提取后，重新修正已有的预测边框，去除无效的边框
            # 使用这个之后，里面的语句不需要进行梯度计算以及反向传播
            with torch.no_grad():
                # 这里的loss需要看一下，这个subsample以及subsample_for_da都是进行改动了的
                #    已经看完了
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        # 从修正后的预测边框对应的特征图里进一步提取特征，这一个步骤就类似于ROI-Pooling+预测前面的线性层
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        # 利用从修正后的预测边框对应的特征进行边框类别和边框偏差信息的预测
        # 这里计算出来的class_logits, box_regression是包含了源域和目标域的，只不过计算loss的时候只提取了源域的
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            # 如果在测试阶段，直接预测图片中的边框（bbox预测的是proposal的变换，所以要进行处理才可以形成框）
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}, x, None

        # 计算loss值
        # 这里调用的是--call--函数
        # 源域数据和目标与数据是混合的，作者在subsample那里，给proposals添加了domain-label属性，
        #   使得在计算faster的loss的时候，可以提取出来源域的数据，只对源域进行计算
        loss_classifier, loss_box_reg, _ = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        # DA start
        if self.training:
            with torch.no_grad():
                # 对于proposal只添加了domain-label，没有添加labels和regression_targets
                #    另外，proposal的label都是0，这影响了正负样本，具体看里面的代码
                da_proposals = self.loss_evaluator.subsample_for_da(proposals, targets)

        # 这里得到的就是instance-level的特征，是经过了线性层的（这里其实是resnet）
        # 这里提取到的da-ins-feas也是有源域和目标域的
        da_ins_feas = self.feature_extractor(features, da_proposals)
        # 这里的predictor和labels没有关系，这里相当于和上面predictor一样，对源域和目标域数据都做了预测
        class_logits, box_regression = self.predictor(da_ins_feas)
        # 需要注意的是，这个domain-labels源域是1，目标域是0
        # 在计算这个loss的时候，使用的proposal还是subsample时候得到的proposal而不是subsample for da的
        #    但是使用的class_logits, box_regression是使用subsample for da预测出来的
        _, _, da_ins_labels = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        # todo li 为什么不在计算源域的loss时候，直接就得到da_ins_labels，而是需要在计算一次loss呢
        #   我目前觉得这是完全没必要的，因为loss.py的--call--函数里面，
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
            da_ins_feas,
            da_ins_labels
        )
        # DA end

def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)
