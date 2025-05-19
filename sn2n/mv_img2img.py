from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Any
from rich.console import Console
import torch
from torch import Tensor, nn
from torch.autograd import Variable
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version
from transformers import DPTImageProcessor, DPTForDepthEstimation

from diffusers import StableDiffusionControlNetImg2ImgPipeline

import sn2n.sa_handler as sa_handler

# SDXL_SOURCE = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_SOURCE = "runwayml/stable-diffusion-v1-5"
CONTROLNET_SOURCE = "diffusers/controlnet-depth-sdxl-1.0-small"
VAE_SOURCE = "madebyollin/sdxl-vae-fp16-fix"

CONSOLE = Console(width=120)

class StylePix2Pix(nn.Module):
    """InstructPix2Pix implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(self, device: Union[torch.device, str], num_train_timesteps: int = 1000, sp2p_use_full_precision=False,
                share_group_norm=True, # False
                share_layer_norm=True, # False
                share_attention=True,
                adain_queries=True,
                adain_keys=True,
                adain_values=False, # False
                full_attention_share=False): # Added) -> None:
        super().__init__()

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.sp2p_use_full_precision = sp2p_use_full_precision

        self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-small-midas").to("cuda")
        self.feature_processor = DPTImageProcessor.from_pretrained("Intel/dpt-small-midas")
        
        # New
        self.controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_SOURCE, #"diffusers/controlnet-depth-sdxl-1.0",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(VAE_SOURCE, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(self.device)

        #pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        # pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            SDXL_SOURCE,
            controlnet=self.controlnet,
            vae=self.vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        )
        assert pipe is not None
        pipe = pipe.to(self.device)

        self.pipe = pipe

        # improve memory performance
        pipe.enable_attention_slicing()

        self.scheduler = pipe.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        pipe.unet.eval()
        pipe.vae.eval()

        # use for improved quality at cost of higher memory
        if self.sp2p_use_full_precision:
            pipe.unet.float()
            pipe.vae.float()
        else:
            if self.device.index:
                #pipe.enable_model_cpu_offload(self.device.index)
                pipe.enable_model_cpu_offload(torch.cuda.current_device())
            else:
                pipe.enable_model_cpu_offload(0)

        self.unet = pipe.unet
        #self.auto_encoder = pipe.vae # remove

        # SA handler
        sa_args = sa_handler.StyleAlignedArgs(share_group_norm=share_group_norm, # False
                                      share_layer_norm=share_layer_norm, # False
                                      share_attention=share_attention,
                                      adain_queries=adain_queries,
                                      adain_keys=adain_keys,
                                      adain_values=adain_values, # False
                                      full_attention_share=full_attention_share, # Added
                                     )
        handler = sa_handler.Handler(pipe)
        handler.register(sa_args, )
        print('sa_args', sa_args)
        #print('WARNING no shared attention!')

        CONSOLE.print("StylePix2Pix loaded!")

    def get_depth_map(self, image: Image, feature_processor: DPTImageProcessor, depth_estimator: DPTForDepthEstimation) -> Image:
        image = feature_processor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = depth_estimator(image).predicted_depth
    
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)
    
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image
    
    def concat_zero_control(self, control_reisduel: T) -> T:
        b = control_reisduel.shape[0] // 2
        zerso_reisduel = torch.zeros_like(control_reisduel[0:1])
        return torch.cat((zerso_reisduel, control_reisduel[:b], zerso_reisduel, control_reisduel[b::]))

    @torch.no_grad()
    def controlnet_call(
        self,
        pipeline: StableDiffusionXLControlNetPipeline,# self.pipe
        prompt: str | list[str] = None,
        prompt_2: str | list[str] | None = None,
        image: PipelineImageInput = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: str | list[str] | None = None,
        negative_prompt_2: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        latents: TN = None,
        prompt_embeds: TN = None,
        negative_prompt_embeds: TN = None,
        pooled_prompt_embeds: TN = None,
        negative_pooled_prompt_embeds: TN = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        controlnet_conditioning_scale: float | list[float] = 1.0,
        control_guidance_start: float | list[float] = 0.0,
        control_guidance_end: float | list[float] = 1.0,
        original_size: tuple[int, int] = None,
        crops_coords_top_left: tuple[int, int] = (0, 0),
        target_size: tuple[int, int] | None = None,
        negative_original_size: tuple[int, int] | None = None,
        negative_crops_coords_top_left: tuple[int, int] = (0, 0),
        negative_target_size:tuple[int, int] | None = None,
        clip_skip: int | None = None,
    ) -> list[Image]:
        
        controlnet = self.controlnet._orig_mod if is_compiled_module(pipeline.controlnet) else self.controlnet
        
        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )
    
        # 1. Check inputs. Raise error if not correct
        pipeline.check_inputs(
            prompt,
            prompt_2,
            image,
            1,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
        )
    
        pipeline._guidance_scale = guidance_scale
    
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
    
        device = pipeline._execution_device
    
        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(
            prompt,
            prompt_2,
            device,
            1,
            True,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
    
        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = pipeline.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=1,
                num_images_per_prompt=1,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=True,
                guess_mode=False,
            )
            height, width = image.shape[-2:]
            image = torch.stack([image[0]] * num_images_per_prompt + [image[1]] * num_images_per_prompt)
        else:
            assert False
        # 5. Prepare timesteps
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipeline.scheduler.timesteps
    
        # 6. Prepare latent variables
        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            1 + num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
            
        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
    
        # 7. Prepare extra step kwargs.
        extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)
    
        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
    
        # 7.2 Prepare added time ids & embeddings
        if isinstance(image, list):
            original_size = original_size or image[0].shape[-2:]
        else:
            original_size = original_size or image.shape[-2:]
        target_size = target_size or (height, width)
    
        add_text_embeds = pooled_prompt_embeds
        if pipeline.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim
    
        add_time_ids = pipeline._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = pipeline._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids
    
        prompt_embeds = torch.stack([prompt_embeds[0]] + [prompt_embeds[1]] * num_images_per_prompt)
        negative_prompt_embeds = torch.stack([negative_prompt_embeds[0]] + [negative_prompt_embeds[1]] * num_images_per_prompt)
        negative_pooled_prompt_embeds = torch.stack([negative_pooled_prompt_embeds[0]] + [negative_pooled_prompt_embeds[1]] * num_images_per_prompt)
        add_text_embeds = torch.stack([add_text_embeds[0]] + [add_text_embeds[1]] * num_images_per_prompt)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
    
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(1 + num_images_per_prompt, 1)
        batch_size = num_images_per_prompt + 1
        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order
        is_unet_compiled = is_compiled_module(pipeline.unet)
        is_controlnet_compiled = is_compiled_module(pipeline.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        controlnet_prompt_embeds = torch.cat((prompt_embeds[1:batch_size], prompt_embeds[1:batch_size]))
        controlnet_added_cond_kwargs = {key: torch.cat((item[1:batch_size,], item[1:batch_size])) for key, item in added_cond_kwargs.items()}
        with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)           
    
                # controlnet(s) inference
                control_model_input = torch.cat((latent_model_input[1:batch_size,], latent_model_input[batch_size+1:]))
    
                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]
                if cond_scale > 0:
                    down_block_res_samples, mid_block_res_sample = pipeline.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=image,
                        conditioning_scale=cond_scale,
                        guess_mode=False,
                        added_cond_kwargs=controlnet_added_cond_kwargs,
                        return_dict=False,
                    )
        
                    mid_block_res_sample = self.concat_zero_control(mid_block_res_sample)
                    down_block_res_samples =  [self.concat_zero_control(down_block_res_sample) for down_block_res_sample in down_block_res_samples]
                else:
                    mid_block_res_sample = down_block_res_samples = None
                # predict the noise residual
                noise_pred = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
    
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
                # compute the previous noisy sample x_t -> x_t-1
                latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
    
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                    progress_bar.update()
                   
        # manually for max memory savings
        if pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast:
            pipeline.upcast_vae()
            latents = latents.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)
    
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast
    
        if needs_upcasting:
            pipeline.upcast_vae()
            latents = latents.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)
    
        image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    
        # cast back to fp16 if needed
        if needs_upcasting:
            pipeline.vae.to(dtype=torch.float16)
     
        if pipeline.watermark is not None:
            image = pipeline.watermark.apply_watermark(image)
    
        #image = pipeline.image_processor.postprocess(image, output_type='pil')
    
        # Offload all models
        pipeline.maybe_free_model_hooks()
        return image

    @torch.no_grad()
    def controlnet_img2img_call(
        self,
        pipeline: StableDiffusionControlNetImg2ImgPipeline,
        prompt: str | list[str] = None,
        prompt_2: str | list[str] | None = None,
        image: PipelineImageInput = None,
        control_image: PipelineIageInput = None,
        height: int | None = None,
        width: int | None = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: str | list[str] | None = None,
        negative_prompt_2: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        latents: TN = None,
        prompt_embeds: TN = None,
        negative_prompt_embeds: TN = None,
        pooled_prompt_embeds: TN = None,
        negative_pooled_prompt_embeds: TN = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        controlnet_conditioning_scale: float | list[float] = 1.0,
        control_guidance_start: float | list[float] = 0.0,
        control_guidance_end: float | list[float] = 1.0,
        original_size: tuple[int, int] = None,
        crops_coords_top_left: tuple[int, int] = (0, 0),
        target_size: tuple[int, int] | None = None,
        negative_original_size: tuple[int, int] | None = None, # need update
        negative_crops_coords_top_left: tuple[int, int] = (0, 0), # need update
        negative_target_size:tuple[int, int] | None = None, # need update
        aesthetic_score: float = 6.0, #added
        negative_aesthetic_score: float = 2.5, #added
        clip_skip: int | None = None,
    ) -> list[Image]:
        controlnet = pipeline.controlnet._orig_mod if is_compiled_module(pipeline.controlnet) else pipeline.controlnet
        #controlnet = pipeline.controlnet
        #controlnet.eval()
        
        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )
    
        # 1. Check inputs. Raise error if not correct
        print('Debug controlnet_conditioning_scale', controlnet_conditioning_scale, print(isinstance(controlnet_conditioning_scale, float)))
        # pipeline.check_inputs(
        #     prompt,
        #     prompt_2,
        #     control_image,
        #     strength,
        #     num_inference_steps,
        #     1,
        #     negative_prompt,
        #     negative_prompt_2,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     pooled_prompt_embeds,
        #     negative_pooled_prompt_embeds,
        #     controlnet_conditioning_scale,
        #     control_guidance_start,
        #     control_guidance_end,
        # )
        
        pipeline.check_inputs(
            prompt=prompt,
            prompt_2=prompt_2,
            image=control_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            callback_steps=1,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
        )
    
        pipeline._guidance_scale = guidance_scale
    
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
    
        device = pipeline._execution_device
    
        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(
            prompt,
            prompt_2,
            device,
            1,
            True,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
    
        # 4. Prepare image
        # 4. Prepare image and controlnet_conditioning_image
        image = pipeline.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        
        if isinstance(controlnet, ControlNetModel):
            control_image = pipeline.prepare_control_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt, # changed from 1
                num_images_per_prompt=1,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=True,
                guess_mode=False,
            )
            height, width = control_image.shape[-2:]
            
            #ref_image = load_image("./temp/cat.png").resize((1024, 1024))
            #ref_image = pipeline.image_processor.preprocess(ref_image, height=height, width=width).to(dtype=torch.float32)
            #image = torch.cat([ref_image, image])
            #image = torch.cat([image, image])
            #image = torch.cat([torch.rand_like(image),image])
            #print('DEBUG control_image before stack', control_image.shape)
            #print('DEBUG', control_image.mean(), control_image.min(), control_image.max())
            #image = torch.stack([image * num_images_per_prompt * 2 + [image[1]] * num_images_per_prompt * 2)
            
            #control_image = torch.stack([control_image[0]] * num_images_per_prompt + [control_image[1]] * num_images_per_prompt) # style-aligned
            
            #print('DEBUG image, control_image', image.shape, control_image.shape) # [2, 3, 1024, 1024], [2, 3, 1024, 1024]
        else:
            assert False
        # 5. Prepare timesteps
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device) 
        #timesteps = pipeline.scheduler.timesteps
        
        # added
        timesteps, num_inference_steps = pipeline.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        #print('DEBUG batch_size * image.shape[0]', batch_size, image.shape[0])
        #latent_timestep = timesteps[:1].repeat(image.shape[0])
        pipeline._num_timesteps = len(timesteps)
        #print('DEBUG timesteps, latent_timestep', timesteps.shape, latent_timestep)
    
        # 6. Prepare latent variables
        num_channels_latents = pipeline.unet.config.in_channels
        # pipeline_controlnet (not img2img) def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # piepline_controlnet_img2img def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        '''
        latents = pipeline.prepare_latents(
            1 + num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        '''
        #print('DEBUG prepare_latents', image.shape, latent_timestep.shape)
        latents = pipeline.prepare_latents(
        #latents = self.prepare_img2img_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt, # 1 + num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
        )                
            
        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
    
        # 7. Prepare extra step kwargs.
        extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)
    
        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
    
        # 7.2 Prepare added time ids & embeddings
        if isinstance(control_image, list):
            original_size = original_size or control_image[0].shape[-2:]
        else:
            original_size = original_size or control_image.shape[-2:]
        target_size = target_size or (height, width)

        # added for img2img
        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size
        add_text_embeds = pooled_prompt_embeds
        
        if pipeline.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim

        #added
        add_time_ids, negative_add_time_ids = pipeline._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)
        
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        negative_add_time_ids = negative_add_time_ids.repeat(batch_size * num_images_per_prompt, 1)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0) # requires debug
        
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device) #.repeat(batch_size * num_images_per_prompt, 1)
        
        #batch_size = num_images_per_prompt + 1 #
        # 8. Denoising loop (Style-Aligned Original)
        num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order
        is_unet_compiled = is_compiled_module(pipeline.unet)
        is_controlnet_compiled = is_compiled_module(pipeline.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")

        with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) # [4,4,128,128]
                latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)     

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    
                # controlnet(s) inference
                control_model_input = latent_model_input # from vanilla
                controlnet_prompt_embeds = prompt_embeds # from vanilla
                controlnet_added_cond_kwargs = added_cond_kwargs # from vanilla

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]
                if cond_scale > 0:
                    #with torch.no_grad():
                    down_block_res_samples, mid_block_res_sample = pipeline.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=control_image,
                        conditioning_scale=cond_scale,
                        guess_mode=False,
                        added_cond_kwargs=controlnet_added_cond_kwargs,
                        return_dict=False,
                    )
        
                else:
                    mid_block_res_sample = down_block_res_samples = None

                noise_pred = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
    
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2) # torch.Size([2, 4, 128, 128]) torch.Size([2, 4, 128, 128])
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
                # compute the previous noisy sample x_t -> x_t-1
                latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
    
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                    progress_bar.update()
                #print('DEBUG END')
                #sys.exit()
                       
        # manually for max memory savings
        if pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast:
            pipeline.upcast_vae()
            latents = latents.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)
    
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast
    
        if needs_upcasting:
            pipeline.upcast_vae()
            latents = latents.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)
    
        image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    
        # cast back to fp16 if needed
        if needs_upcasting:
            pipeline.vae.to(dtype=torch.float16)
     
        if pipeline.watermark is not None:
            image = pipeline.watermark.apply_watermark(image)
    
        #image = pipeline.image_processor.postprocess(image, output_type='pil')
    
        # Offload all models
        pipeline.maybe_free_model_hooks()
        return image


    def latents_to_img(self, latents: Float[Tensor, "BS 4 H W"]) -> Float[Tensor, "BS 3 H W"]:
        """Convert latents to images
        Args:
            latents: Latents to convert
        Returns:
            Images
        """

        latents = 1 / CONST_SCALE * latents

        with torch.no_grad():
            imgs = self.auto_encoder.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def imgs_to_latent(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert images to latents
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        posterior = self.auto_encoder.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents

    def prepare_image_latents(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert conditioning image to latents used for classifier-free guidance
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        image_latents = self.auto_encoder.encode(imgs).latent_dist.mode()

        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
