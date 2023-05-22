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

import time

import torch
import torch.nn as nn

from ..backbone import build_backbone
from ..fpn import build_fpn
from ..head import build_head


class OneStageDetectorSegmentor(nn.Module):
    def __init__(
        self,
        backbone_cfg,
        fpn_cfg=None,
        head_cfg=None,
        msk_cfg=None
    ):
        super(OneStageDetectorSegmentor, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        if fpn_cfg is not None:
            self.fpn = build_fpn(fpn_cfg)
        if head_cfg is not None:
            self.head = build_head(head_cfg)
        if msk_cfg is not None:
            self.mask = msk_cfg
        self.epoch = 0

    def forward(self, x):
        x = self.backbone(x)
        if hasattr(self, "fpn"):
            features = self.fpn(x)
        if hasattr(self, "head"):
            x = self.head(features)
        return x,features

    def inference(self, meta):
        with torch.no_grad():
            torch.cuda.synchronize()
            time1 = time.time()
            # features are for mask prediction
            preds,features = self(meta["img"])
            torch.cuda.synchronize()
            time2 = time.time()
            print("forward time: {:.3f}s".format((time2 - time1)), end=" | ")
            masks = self.head.masks_process(preds,features,meta)
            results = self.head.post_process(preds,masks, meta)
            torch.cuda.synchronize()
            print("decode time: {:.3f}s".format((time.time() - time2)), end=" | ")
        return results

    def forward_train(self, gt_meta):
        preds,features = self(gt_meta["img"])
        #masks = self.head.masks_process(preds,features,gt_meta)
        _,mask_loss=self.head.process_mask_train(preds,features,gt_meta)
        loss, loss_states = self.head.loss(preds, gt_meta)
        # Add mask loss to loss dict
        loss+=mask_loss
        loss_states["MaskLoss"]=mask_loss
        return preds, loss, loss_states


    def set_epoch(self, epoch):
        self.epoch = epoch
