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
Style-NeRF2NeRF configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig

from sn2n.sn2n_datamanager import StyleNeRF2NeRFDataManagerConfig
from sn2n.sn2n_dataparser import NerfstudioData2ParserConfig # added custom parser
from sn2n.sn2n import StyleNeRF2NeRFModelConfig
from sn2n.sn2n_trainer import StyleNeRF2NeRFTrainerConfig
from sn2n.sn2n_pipeline import StyleNeRF2NeRFPipelineConfig

from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig


sn2n_method = MethodSpecification(
    config=StyleNeRF2NeRFTrainerConfig(
        method_name="sn2n",
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=250,
        max_num_iterations=15000, # default 15000
        save_only_latest_checkpoint=True,
        mixed_precision=True,
        pipeline=StyleNeRF2NeRFPipelineConfig(
            datamanager=StyleNeRF2NeRFDataManagerConfig(
            #datamanager=ParallelDataManagerConfig(
                dataparser=NerfstudioData2ParserConfig(),
                #dataparser=NerfstudioDataParserConfig(), # rollback to this when you encouter problems
                #dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=256*256,
                eval_num_rays_per_batch=256*256,
                patch_size=256
            ),
            model=StyleNeRF2NeRFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_swd=True,
                #camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=5000),
            },
            #"encodings": {
            #    "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            #    "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.002, max_steps=30000),
            #},
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Style-NeRF2NeRF primary method: uses LPIPS at full precision",
)