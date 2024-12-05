# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""
Model for StyleNeRF2NeRF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

#from typing import Type
from typing import Optional, Type

import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    interlevel_loss,
)
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.models.tensorf import TensoRFModel, TensoRFModelConfig

#from sn2n.sp2p import StylePix2Pix
import torchvision.transforms.functional as F
from sn2n.swdloss import VGG19

import sys
from PIL import Image
import torchvision.transforms.functional as F

@dataclass
class StyleNeRF2NeRFModelConfig(NerfactoModelConfig):
    """Configuration for the StyleNeRF2NeRFModel."""
    _target: Type = field(default_factory=lambda: StyleNeRF2NeRFModel)
    use_swd: bool = True
    """Whether to use SWD loss"""
    use_l1: bool = True
    """Whether to use L1 loss"""
    patch_size: int = 256 #128 sshould be the same as patch_size in patch-based sampling
    """Patch size to use."""
    swd_loss_mult: float = 1.0
    """Multiplier for LPIPS loss."""
    rgb_loss_mult: float = 1.0
    """Multiplier for RGB loss."""
    blend_weight: float = 0.5
    """Weight for style blending."""

class StyleNeRF2NeRFModel(NerfactoModel):
    """Model for StyleNeRF2NeRF."""

    config: StyleNeRF2NeRFModelConfig

    def populate_modules(self):
        """Required to use L1 Loss."""
        super().populate_modules()

        if self.config.use_l1:
            self.rgb_loss = L1Loss()
        else:
            self.rgb_loss = MSELoss()
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        vgg = VGG19().to(torch.device("cuda"))
        vgg.load_state_dict(torch.load("vgg19.pth")) # Load VGG19 weights
        self.vgg = vgg

        print('SWD Patch size:', self.config.patch_size)

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        blend_images = 'image_second' in batch.keys()
        use_tags = 'tag' in batch.keys()
        if blend_images:
            image_second = batch["image_second"].to(self.device)
        if use_tags:
            tag = batch["tag"].to(self.device)*1.
            tag_patches = tag.view(-1, self.config.patch_size,self.config.patch_size, 1).permute(0, 3, 1, 2)
            second_patches = (image_second.view(-1, self.config.patch_size, self.config.patch_size, 3).permute(0, 3, 1, 2)).clamp(0, 1)

        if self.config.use_swd:
            # outputs['rgb'].shape: [num_of_rays, 3]
            if self.training:

                out_patches = (outputs["rgb"].view(-1, self.config.patch_size, self.config.patch_size, 3).permute(0, 3, 1, 2)).clamp(0, 1) # -1 1
                gt_patches = (image.view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2)).clamp(0, 1)

                #out_patches = (outputs["rgb"].view(-1, self.config.patch_size, 384, 3).permute(0, 3, 1, 2)).clamp(-1, 1)
                #gt_patches = (image.view(-1, self.config.patch_size, 384, 3).permute(0, 3, 1, 2)).clamp(-1, 1)

                if blend_images:
                    gt_patches_second = (image_second.view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)
                #if use_tags:
                #    out_patches = out_patches*tag_patches
                #    gt_patches = gt_patches*tag_patches
            else:
                out_patches = (outputs["rgb"].view(-1, self.config.patch_size, self.config.patch_size, 3).permute(0, 3, 1, 2)).clamp(0, 1) # -1 1
                gt_patches = (image.view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2)).clamp(0, 1)

            if use_tags: # Experimental!!

                #gt_patches = gt_patches*tag_patches + second_patches*(1-tag_patches) # testing region-based
                #out_patches = out_patches*tag_patches
                #gt_patches = gt_patches*tag_patches

                loss_dict["swd_loss"] = self.config.swd_loss_mult*self.vgg.ebsw_loss(out_patches, gt_patches, mask=tag_patches) # [b, c, h, w]
                loss_dict["swd_loss"] += 0.1*self.vgg.content_loss(out_patches, second_patches)

            elif blend_images:
                loss_dict["swd_loss"] = self.config.blend_weight*self.vgg.slicing_loss(out_patches, gt_patches) + (1.-self.config.blend_weight)*self.vgg.slicing_loss(out_patches, gt_patches_second) # Wasserstein Barycenter
                #loss_dict["swd_loss"] = self.vgg.slicing_loss(out_patches, gt_patches_second) # Wasserstein Barycenter
            else:
                loss_dict["swd_loss"] = self.config.swd_loss_mult*self.vgg.slicing_loss(out_patches, gt_patches)
                loss_dict["swd_loss"] += 0.1*self.vgg.content_loss(out_patches, gt_patches)
                #loss_dict["swd_loss"] = self.lpips(out_patches, gt_patches) # for ablation only
                #loss_dict["swd_loss"] = self.vgg.gram_loss(out_patches, gt_patches) # for ablation only

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
        
        return loss_dict
