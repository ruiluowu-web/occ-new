import torch
from diffusers.pipelines import FluxPipeline
from typing import List, Union, Optional, Dict, Any, Callable
import numpy as np
from .tools import Layout, encode_layout_flux
import os
from PIL import Image
import matplotlib.pyplot as plt

from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
)


def save_predicted_masks(pipeline, output_dir: str, height: int, width: int, timestep_key: str, layer_masks: List[Dict], bbox_masks=None):
    os.makedirs(output_dir, exist_ok=True)
    timestep_dir = os.path.join(output_dir, timestep_key)
    os.makedirs(timestep_dir, exist_ok=True)

    for mask_data in layer_masks:
        layer_idx = mask_data['layer_idx']
        mask_probs = mask_data['mask_probs']
        seq_len = mask_probs.shape[1]
        img_size = int(seq_len ** 0.5)

        for obj_idx in range(mask_probs.shape[0]):
            mask_2ch = mask_probs[obj_idx]
            mask_2ch_2d = mask_2ch.reshape(img_size, img_size, 2)

            mask_2ch_2d = mask_2ch_2d.permute(2, 0, 1)
            mask_2ch_upsampled = torch.nn.functional.interpolate(
                mask_2ch_2d.unsqueeze(0),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            background_prob = mask_2ch_upsampled[0]
            foreground_prob = mask_2ch_upsampled[1]
            mask_upsampled = foreground_prob

            if bbox_masks is not None and obj_idx < len(bbox_masks):
                bbox_mask = bbox_masks[obj_idx]

                if bbox_mask.dim() == 1:
                    bbox_size = int(bbox_mask.numel() ** 0.5)
                    bbox_mask_2d = bbox_mask.reshape(bbox_size, bbox_size)
                elif bbox_mask.dim() == 2:
                    if bbox_mask.shape[1] == 1:
                        bbox_size = int(bbox_mask.shape[0] ** 0.5)
                        bbox_mask_2d = bbox_mask.squeeze(-1).reshape(bbox_size, bbox_size)
                    else:
                        bbox_mask_2d = bbox_mask
                else:
                    bbox_mask_2d = bbox_mask.squeeze()

                if bbox_mask_2d.shape != mask_upsampled.shape:
                    bbox_mask_2d = torch.nn.functional.interpolate(
                        bbox_mask_2d.unsqueeze(0).unsqueeze(0).float(),
                        size=(height, width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()

                mask_upsampled = mask_upsampled * bbox_mask_2d

            mask_upsampled = mask_upsampled.to(torch.float32).numpy()
            mask_img = (mask_upsampled * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(
                os.path.join(timestep_dir, f"layer_{layer_idx}_obj_{obj_idx}_mask.png")
            )
            

def prepare_params(
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    **kwargs: dict,
):
    return (
        prompt,
        prompt_2,
        height,
        width,
        num_inference_steps,
        timesteps,
        guidance_scale,
        num_images_per_prompt,
        generator,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        output_type,
        return_dict,
        joint_attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
    )


@torch.no_grad()
def inference(
    pipeline: FluxPipeline,
    layout_transformer = None,
    layout: Optional[Layout] = None,
    enable_layout: bool = True,
    grounding_ratio: float = 1.0,
    **params: dict,
):

    self = pipeline
    (
        prompt,
        prompt_2,
        height,
        width,
        num_inference_steps,
        timesteps,
        guidance_scale,
        num_images_per_prompt,
        generator,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        output_type,
        return_dict,
        joint_attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
    ) = prepare_params(**params)

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None)
        if self.joint_attention_kwargs is not None
        else None
    )
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    num_channels_latents = layout_transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    use_condition = layout is not None
    if use_condition:
        latent_h = int(height) // self.vae_scale_factor // 2
        latent_w = int(width) // self.vae_scale_factor // 2
        layout_kwargs = encode_layout_flux(layout, self, (latent_h, latent_w))
        
        if hasattr(layout, 'occluder') and hasattr(layout, 'bbox_masks') and layout.occluder is not None and layout.bbox_masks is not None:
            from diffusers.pipelines import FluxPipeline
            
            bbox_masks = layout.bbox_masks
            bbox_masks_downsampled = torch.nn.functional.interpolate(
                bbox_masks.unsqueeze(1).to(device=device, dtype=prompt_embeds.dtype),
                size=(latent_h*2, latent_w*2),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
            packed_bbox_masks = []
            for layout_idx in range(bbox_masks_downsampled.shape[0]):
                mask_2d = bbox_masks_downsampled[layout_idx]
                mask_packed = FluxPipeline._pack_latents(
                    mask_2d.unsqueeze(0).unsqueeze(0),
                    batch_size=1,
                    num_channels_latents=1,
                    height=latent_h*2,
                    width=latent_w*2,
                )
                packed_bbox_masks.append(mask_packed.squeeze(0)[..., 0:1])
            
            layout_kwargs["layout"]["occlusion"] = layout.occluder
            layout_kwargs["layout"]["bbox_mask"] = packed_bbox_masks
            layout_kwargs["layout"]["img_height"] = latent_h
            layout_kwargs["layout"]["img_width"] = latent_w
        
        enable_layout = bool(enable_layout)
    else:
        layout_kwargs = None
        enable_layout = False

    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    if getattr(self.scheduler.config, "use_flow_sigmas", False):
        sigmas = None
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * self.scheduler.order, 0
    )
    self._num_timesteps = len(timesteps)

    num_grounding_steps = int(grounding_ratio * len(timesteps))

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            if i == num_grounding_steps:
                enable_layout = False

            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            if layout_transformer.config.guidance_embeds:
                guidance = torch.tensor([guidance_scale], device=device)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None
            
            transformer_output = layout_transformer(
                layout_kwargs=layout_kwargs,
                enable_layout=enable_layout,
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=True,
            )

            noise_pred = transformer_output.sample
            mask_pred = transformer_output.z_buffer_masks if hasattr(transformer_output, 'z_buffer_masks') else None
            
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

    if output_type == "latent":
        image = latents

    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    self.maybe_free_model_hooks()
    
    if not return_dict:
        return (image,)

    output = FluxPipelineOutput(images=image)
    
    return output


@torch.no_grad()
def inference_edit(
    pipeline: FluxPipeline,
    layout_transformer,
    init_image,
    layout: Optional[Layout] = None,
    edit_mask: Optional[torch.Tensor] = None,
    enable_layout: bool = True,
    grounding_ratio: float = 1.0,
    edit_strength: float = 0.9,
    seed: int = 0,
    **params,
):
    """Flow-matching inpainting edit with per-step trajectory blending.

    Unmasked regions follow the exact original ODE trajectory:
        z_known_t = (t/N) * z_1 + (1 - t/N) * z_0
    Masked regions are regenerated by the model with filtered layout.

    Args:
        edit_mask: (H, W) float32 tensor, 1.0 = regenerate this pixel.
                   Computed as the union of removed instance bboxes.
    """
    self = pipeline

    (
        prompt, prompt_2, height, width,
        num_inference_steps, timesteps, guidance_scale,
        num_images_per_prompt, generator, latents,
        prompt_embeds, pooled_prompt_embeds,
        output_type, return_dict,
        joint_attention_kwargs,
        callback_on_step_end, callback_on_step_end_tensor_inputs,
        max_sequence_length,
    ) = prepare_params(**params)

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    self.check_inputs(
        prompt, prompt_2, height, width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    batch_size = 1
    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None)
        if self.joint_attention_kwargs is not None
        else None
    )
    (prompt_embeds, pooled_prompt_embeds, text_ids) = self.encode_prompt(
        prompt=prompt, prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    vae_scale = self.vae_scale_factor
    latent_h = 2 * (int(height) // (vae_scale * 2))
    latent_w = 2 * (int(width) // (vae_scale * 2))

    if isinstance(init_image, Image.Image):
        img_tensor = self.image_processor.preprocess(init_image)
    else:
        img_tensor = init_image
    img_tensor = img_tensor.to(device=device, dtype=self.dtype)
    z_0_decoded = self.vae.encode(img_tensor).latent_dist.mode()
    z_0_raw = (z_0_decoded - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    num_channels_latents = layout_transformer.config.in_channels // 4
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (batch_size, num_channels_latents, latent_h, latent_w)
    z_1_unpacked = torch.randn(shape, generator=g, device=device, dtype=prompt_embeds.dtype)

    z_1 = self._pack_latents(z_1_unpacked, batch_size, num_channels_latents, latent_h, latent_w)
    z_0_packed = self._pack_latents(z_0_raw, batch_size, num_channels_latents, latent_h, latent_w)

    packed_mask = None
    if edit_mask is not None:
        mask_tensor = edit_mask.to(device=device, dtype=torch.float32)
        mask_ds = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0),
            size=(latent_h, latent_w),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0).squeeze(0)
        mask_4ch = mask_ds.unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1)
        packed_mask = self._pack_latents(
            mask_4ch, batch_size=1, num_channels_latents=4,
            height=latent_h, width=latent_w,
        )
        packed_mask = packed_mask[:, :, :1]
        print(f"[EDIT DEBUG] packed_mask shape={packed_mask.shape}, "
              f"min={packed_mask.min().item():.3f}, max={packed_mask.max().item():.3f}, "
              f"mean={packed_mask.mean().item():.3f}, "
              f"sum>0.5={(packed_mask > 0.5).sum().item()}")

    latent_image_ids = self._prepare_latent_image_ids(
        batch_size, latent_h // 2, latent_w // 2, device, prompt_embeds.dtype
    )

    use_condition = layout is not None
    if use_condition:
        latent_h_layout = int(height) // vae_scale // 2
        latent_w_layout = int(width) // vae_scale // 2
        layout_kwargs = encode_layout_flux(layout, self, (latent_h_layout, latent_w_layout))

        if (hasattr(layout, 'occluder') and hasattr(layout, 'bbox_masks')
                and layout.occluder is not None and layout.bbox_masks is not None):
            bbox_masks = layout.bbox_masks
            bbox_masks_ds = torch.nn.functional.interpolate(
                bbox_masks.unsqueeze(1).to(device=device, dtype=prompt_embeds.dtype),
                size=(latent_h_layout * 2, latent_w_layout * 2),
                mode='bilinear', align_corners=False
            ).squeeze(1)

            packed_bbox_masks = []
            for layout_idx in range(bbox_masks_ds.shape[0]):
                mask_2d = bbox_masks_ds[layout_idx]
                mask_packed = FluxPipeline._pack_latents(
                    mask_2d.unsqueeze(0).unsqueeze(0),
                    batch_size=1, num_channels_latents=1,
                    height=latent_h_layout * 2, width=latent_w_layout * 2,
                )
                packed_bbox_masks.append(mask_packed.squeeze(0)[..., 0:1])

            layout_kwargs["layout"]["occlusion"] = layout.occluder
            layout_kwargs["layout"]["bbox_mask"] = packed_bbox_masks
            layout_kwargs["layout"]["img_height"] = latent_h_layout
            layout_kwargs["layout"]["img_width"] = latent_w_layout

        enable_layout = bool(enable_layout)
    else:
        layout_kwargs = None
        enable_layout = False

    image_seq_len = z_0_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    if getattr(self.scheduler.config, "use_flow_sigmas", False):
        sigmas = None
    full_timesteps, _ = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, timesteps, sigmas, mu=mu,
    )

    t_target = edit_strength * self.scheduler.config.num_train_timesteps
    start_idx = int((full_timesteps > t_target).int().sum().item())
    start_idx = min(start_idx, len(full_timesteps) - 1)
    timesteps = full_timesteps[start_idx:]

    t_0_val = timesteps[0].item()
    t_0_norm = t_0_val / self.scheduler.config.num_train_timesteps
    z_start = t_0_norm * z_1 + (1.0 - t_0_norm) * z_0_packed
    latents = z_start.to(dtype=prompt_embeds.dtype)

    if packed_mask is not None:
        print(f"[EDIT DEBUG] t_0_val={t_0_val:.1f}, start_idx={start_idx}, "
              f"edit_steps={len(timesteps)}, t_range=[{timesteps[0].item():.0f},{timesteps[-1].item():.0f}]")

    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    num_grounding_steps = int(grounding_ratio * len(timesteps))

    self.scheduler._step_index = start_idx

    if packed_mask is not None:
        z_init_diff = (latents - z_0_packed).abs().mean().item()
        print(f"[EDIT DEBUG] z_start vs z_0 abs_diff={z_init_diff:.6f}, "
              f"z_start std={latents.std().item():.3f}, z_0 std={z_0_packed.std().item():.3f}")

    with self.progress_bar(total=len(timesteps)) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            if i == num_grounding_steps:
                enable_layout = False

            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            if layout_transformer.config.guidance_embeds:
                guidance = torch.tensor([guidance_scale], device=device)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None

            transformer_output = layout_transformer(
                layout_kwargs=layout_kwargs,
                enable_layout=enable_layout,
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=True,
            )

            noise_pred = transformer_output.sample
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

            if packed_mask is not None:
                t_next = timesteps[i + 1].item() if i + 1 < len(timesteps) else 0.0
                t_next_norm = t_next / self.scheduler.config.num_train_timesteps
                z_known = t_next_norm * z_1 + (1.0 - t_next_norm) * z_0_packed
                mask_blend = packed_mask.to(dtype=latents.dtype)
                if i == 0:
                    diff_before = (latents - z_known).abs().mean().item()
                latents = latents * mask_blend + z_known * (1.0 - mask_blend)
                if i == 0:
                    diff_after = (latents - z_known).abs().mean().item()
                    print(f"[EDIT DEBUG] step0: diff_before_blend={diff_before:.6f}, "
                          f"diff_after_blend={diff_after:.6f}, "
                          f"mask_max={mask_blend.max().item():.3f}")
                if i == len(timesteps) - 1:
                    final_diff_masked = ((latents - z_known) * mask_blend).abs().mean().item()
                    final_diff_unmasked = ((latents - z_known) * (1 - mask_blend)).abs().mean().item()
                    print(f"[EDIT DEBUG] final step: diff_masked={final_diff_masked:.6f}, "
                          f"diff_unmasked={final_diff_unmasked:.6f}")

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, vae_scale)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return FluxPipelineOutput(images=image)
