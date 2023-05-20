# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torchvision.ops import roi_align


from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from .gfl_head import GFLHead


class NanoDetSegmHead(GFLHead):
    """
    Modified from GFL, use same loss functions but much lightweight convolution heads
    """

    def __init__(
        self,
        num_classes,
        loss,
        input_channel,
        stacked_convs=2,
        octave_base_scale=5,
        conv_type="DWConv",
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        reg_max=16,
        share_cls_reg=False,
        activation="LeakyReLU",
        feat_channels=256,
        strides=[8, 16, 32],
        **kwargs
    ):
        self.share_cls_reg = share_cls_reg
        self.activation = activation
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule
        super(NanoDetSegmHead, self).__init__(
            num_classes,
            loss,
            input_channel,
            feat_channels,
            stacked_convs,
            octave_base_scale,
            strides,
            conv_cfg,
            norm_cfg,
            reg_max,
            **kwargs
        )

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for _ in self.strides:
            cls_convs, reg_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        # Segmentation head is shared 
        self.seg_convs=self._build_seg_head()

        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels,
                    self.cls_out_channels + 4 * (self.reg_max + 1)
                    if self.share_cls_reg
                    else self.cls_out_channels,
                    1,
                    padding=0,
                )
                for _ in self.strides
            ]
        )
        # TODO: if
        self.gfl_reg = nn.ModuleList(
            [
                nn.Conv2d(self.feat_channels, 4 * (self.reg_max + 1), 1, padding=0)
                for _ in self.strides
            ]
        )
        # Segmentation Head
        self.segm = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1, padding=0)


    def _build_seg_head(self):
        modules = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            modules.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
        return nn.Sequential(*modules)



    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
            if not self.share_cls_reg:
                reg_convs.append(
                    self.ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None,
                        activation=self.activation,
                    )
                )

        return cls_convs, reg_convs

    def masks_process(self, preds, features, meta):
        cls_scores, bbox_preds = preds.split([self.num_classes, 4 * (self.reg_max + 1)], dim=-1)
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
        input_height, input_width = meta["img"].shape[2:]
        feature_idx = 0
        _, _, fh, fw = features[feature_idx].shape
        spatial_scale = fh/input_height
        all_pred_masks = []
        all_boxes = []
        for i, result in enumerate(result_list):
            image_boxes = result[0]
            if image_boxes.numel() > 0:
                boxes = image_boxes[:,:4]
                output_size = (14, 14)
                aligned_features = roi_align(features[feature_idx][i].unsqueeze(0), [boxes], output_size, spatial_scale=spatial_scale, sampling_ratio=-1)
                # use Sequential here
                pred_masks = self.seg_convs(aligned_features)
                pred_masks = self.segm(pred_masks).sigmoid()
            else:
                pred_masks = torch.tensor([])
                boxes = torch.tensor([])

            all_pred_masks.append(pred_masks)
            all_boxes.append(boxes)

        return all_pred_masks


    def crop_gt_masks(self, gt_mask, boxes):
        """
        gt_mask: A tensor of shape (H, W) representing a binary ground truth mask.
        boxes: A tensor of shape (N, 4) representing predicted bounding boxes, where N is the number of boxes.
        """
        cropped_masks = []
        h, w = gt_mask.shape
        for box in boxes:
            x1, y1, x2, y2 = box.int()
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            cropped_mask = gt_mask[y1:y2, x1:x2]
            cropped_masks.append(cropped_mask.unsqueeze(0))  # Add a dimension for stacking
        cropped_masks = torch.cat(cropped_masks, dim=0)  # Stacks masks along a new dimension
        return cropped_masks

    def crop_and_resize_masks(self, gt_mask, boxes, size):
        cropped_masks = self.crop_gt_masks(gt_mask, boxes)  # Use the crop_gt_masks function defined earlier
        resized_masks = torch.nn.functional.interpolate(cropped_masks, size=size, mode='nearest')
        return resized_masks


    def process_mask_train(self, preds, features, meta):
        cls_scores, bbox_preds = preds.split([self.num_classes, 4 * (self.reg_max + 1)], dim=-1)
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
        gt_masks = meta["gt"]

        input_height, input_width = meta["img"].shape[2:]
        feature_idx = 0
        _, _, fh, fw = features[feature_idx].shape
        spatial_scale = fh/input_height

        all_pred_masks = []
        all_boxes = []
        mask_losses = []

        for i, result in enumerate(result_list):
            image_boxes = result[0]
            if image_boxes.numel() > 0:
                boxes = image_boxes[:,:4]
                output_size = (14, 14)  # The size of the predicted masks
                aligned_features = roi_align(features[feature_idx][i].unsqueeze(0), [boxes], output_size, spatial_scale=spatial_scale, sampling_ratio=-1)
                pred_masks = self.seg_convs(aligned_features)
                pred_masks = self.segm(pred_masks)

                # Crop and resize the ground truth masks based on the predicted boxes
                gt_resized_masks = self.crop_and_resize_masks(gt_masks[i], boxes, pred_masks.shape[-2:])

                # Calculate the mask loss
                mask_loss = self.calculate_mask_loss(pred_masks, gt_resized_masks)
                mask_losses.append(mask_loss)
            else:
                pred_masks = torch.tensor([])
                boxes = torch.tensor([])

            all_pred_masks.append(pred_masks)
            all_boxes.append(boxes)

        return all_pred_masks, torch.stack(mask_losses).mean()  # Return mean mask loss across all images



    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.seg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)

    def forward(self, feats):
        if torch.onnx.is_in_onnx_export():
            return self._forward_onnx(feats)
        outputs = []
        for x, cls_convs, reg_convs, gfl_cls, gfl_reg in zip(feats, self.cls_convs, self.reg_convs, self.gfl_cls, self.gfl_reg):
            cls_feat = x
            reg_feat = x
            for cls_conv in cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in reg_convs:
                reg_feat = reg_conv(reg_feat)
            if self.share_cls_reg:
                output = gfl_cls(cls_feat)
            else:
                cls_score = gfl_cls(cls_feat)
                bbox_pred = gfl_reg(reg_feat)
                output = torch.cat([cls_score, bbox_pred], dim=1)
            outputs.append(output.flatten(start_dim=2))
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs

    def _forward_onnx(self, feats):
        """only used for onnx export"""
        outputs = []
        for x, cls_convs, reg_convs, gfl_cls, gfl_reg in zip(
            feats, self.cls_convs, self.reg_convs, self.gfl_cls, self.gfl_reg
        ):
            cls_feat = x
            reg_feat = x
            for cls_conv in cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in reg_convs:
                reg_feat = reg_conv(reg_feat)
            if self.share_cls_reg:
                output = gfl_cls(cls_feat)
                cls_pred, reg_pred = output.split(
                    [self.num_classes, 4 * (self.reg_max + 1)], dim=1
                )
            else:
                cls_pred = gfl_cls(cls_feat)
                reg_pred = gfl_reg(reg_feat)
            cls_pred = cls_pred.sigmoid()
            out = torch.cat([cls_pred, reg_pred], dim=1)
            outputs.append(out.flatten(start_dim=2))
        return torch.cat(outputs, dim=2).permute(0, 2, 1)
