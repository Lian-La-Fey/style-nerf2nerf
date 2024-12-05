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
Style-NeRF2NeRF Datamanager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

from rich.progress import Console

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
'''
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
    FullImageDatamanager,
)'''
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager

from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path
from nerfstudio.data.datasets.base_dataset import InputDataset
from copy import deepcopy
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import sys

CONSOLE = Console(width=120)

class Sn2nDataset(InputDataset):
    """A trivial dataset with blank images for the viewer"""

    def get_numpy_image_2(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.metadata['image_second_filenames'][image_idx]
        pil_image = Image.open(image_filename)
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.Resampling.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_image_float32_2(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image in float32 torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image_2(image_idx).astype("float32") / 255.0)
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                self._dataparser_outputs.alpha_color <= 1
            ).all(), "alpha color given is out of range between [0, 1]."
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        return image

    def get_image_uint8_2(self, image_idx: int) -> UInt8[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image in uint8 torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image_2(image_idx))
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                self._dataparser_outputs.alpha_color <= 1
            ).all(), "alpha color given is out of range between [0, 1]."
            image = image[:, :, :3] * (image[:, :, -1:] / 255.0) + 255.0 * self._dataparser_outputs.alpha_color * (
                1.0 - image[:, :, -1:] / 255.0
            )
            image = torch.clamp(image, min=0, max=255).to(torch.uint8)
        return image

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
        """
        if image_type == "float32":
            image = self.get_image_float32(image_idx)
        elif image_type == "uint8":
            image = self.get_image_uint8(image_idx)
        else:
            raise NotImplementedError(f"image_type (={image_type}) getter was not implemented, use uint8 or float32")

        data = {"image_idx": image_idx, "image": image}
        if self._dataparser_outputs.mask_filenames is not None:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        #if self.mask_color:
        #    data["image"] = torch.where(
        #        data["mask"] == 1.0, data["image"], torch.ones_like(data["image"]) * torch.tensor(self.mask_color)
        #    )
        if self._dataparser_outputs.metadata['image_second_filenames'] is not None:
            image_second_filepath = self._dataparser_outputs.metadata['image_second_filenames'][image_idx]
            if image_type == "float32":
                image_second = self.get_image_float32_2(image_idx)
            elif image_type == "uint8":
                image_second = self.get_image_uint8_2(image_idx)
            data["image_second"] = image_second
        if self._dataparser_outputs.metadata['tag_filenames'] is not None:
            tag_filepath = self._dataparser_outputs.metadata['tag_filenames'][image_idx]
            data["tag"] = get_image_mask_tensor_from_path(filepath=tag_filepath, scale_factor=self.scale_factor)
            assert (
                data["tag"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['tag'].shape[:2]} and {data['image'].shape[:2]}"            

        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

@dataclass
class StyleNeRF2NeRFDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for the StyleNeRF2NeRFDataManager."""

    _target: Type = field(default_factory=lambda: StyleNeRF2NeRFDataManager)
    #_target: Type = field(default_factory=lambda: ParallelDataManager)
    patch_size: int = 256 # default value, gets overridden by sn2n_config
    """Size of patch to sample from. If >1, patch-based sampling will be used."""
    data2: str = None

class StyleNeRF2NeRFDataManager(VanillaDataManager):
    """Data manager for StyleNeRF2NeRF."""

    config: StyleNeRF2NeRFDataManagerConfig

    def setup_train(self):
        print('setup custom DataManager')

        """Sets up the data loaders for training (Copy from VanillaDataManager.setup_train)"""
        print('DEBUG: overriding dataset type')
        self.train_dataset = Sn2nDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )
        #override
        print('DEBUG train_dataset keys:', self.train_dataset.__getitem__(2).keys())

        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")

        print('train dataset metadata:')
        print(self.train_dataset.metadata) 
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_pixel_sampler.config.patch_size = self.config.patch_size
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))

        # pre-fetch the image batch (how images are replaced in dataset)
        self.image_batch = next(self.iter_train_image_dataloader)
        print('self.image_batch', self.image_batch.keys())
        print('pixel sampler', self.train_pixel_sampler.config)

        # load second data (if any) for style blending
        if self.config.data2 is not None:
            print('DEBUG: data2 provided', self.config.data2)
            #sys.exit()
    
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        self.train_count += 1
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(self.image_batch) # samples pixels (or patches) from sampled image batch
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        return ray_bundle, batch
