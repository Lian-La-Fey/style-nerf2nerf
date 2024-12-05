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

"""StylePix2Pix Pipeline and trainer"""

from dataclasses import dataclass, field
from itertools import cycle
from typing import Optional, Type
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.server.viewer_elements import ViewerNumber, ViewerText

import torchvision.transforms.functional as F

from sn2n.sn2n_datamanager import (
    StyleNeRF2NeRFDataManagerConfig,
)
import einops as eo
import wandb

@dataclass
class StyleNeRF2NeRFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: StyleNeRF2NeRFPipeline)
    """target class to instantiate"""
    #datamanager: StyleNeRF2NeRFDataManagerConfig = StyleNeRF2NeRFDataManagerConfig()
    """specifies the datamanager config"""

class StyleNeRF2NeRFPipeline(VanillaPipeline):
    """StyleNeRF2NeRF pipeline"""

    config: StyleNeRF2NeRFPipelineConfig

    def __init__(
        self,
        config: StyleNeRF2NeRFPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
        
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

        # keep track of spot in dataset
        
        if self.datamanager.config.train_num_images_to_sample_from == -1:
            self.train_indices_order = cycle(range(len(self.datamanager.train_dataparser_outputs.image_filenames)))
        else:
            self.train_indices_order = cycle(range(self.datamanager.config.train_num_images_to_sample_from))

    def get_train_loss_dict_base(self, step: int):
            """This function gets your training loss dict and performs image editing.
            Args:
                step: current iteration step to update sampler if using DDP (distributed)
            """

            ray_bundle, batch = self.datamanager.next_train(step) #
            model_outputs = self.model(ray_bundle)
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

            return model_outputs, loss_dict, metrics_dict
    
    def get_train_loss_dict(self, step: int):
            """
            Args:
                step: current iteration step to update sampler if using DDP (distributed)
            """

            ray_bundle, batch = self.datamanager.next_train(step) #
            #camera, batch = self.datamanager.next_train(step)

            model_outputs = self.model(ray_bundle)

            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

            return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
