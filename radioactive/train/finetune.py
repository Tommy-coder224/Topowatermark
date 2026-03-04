"""
LoRA 微调管线 — 放射性验证的核心实验组件
==========================================

目标：验证定理 D（拓扑放射性）。

实验因果链：
  1. 用户 u 的水印图像 {x_i}^N → 微调模型 M₁ (LoRA)
  2. M₁ 生成新图像 {x'_j}^M
  3. 检测 {x'_j} 中是否保留了用户 u 的拓扑指纹
  4. 若检测率 >> 随机基线 → 定理 D 成立 → 水印具有放射性

技术实现：
  使用 PEFT (Parameter-Efficient Fine-Tuning) 的 LoRA
  对 UNet 的 attention 层做低秩适配。
  LoRA 不改变 VAE → 不引入额外变量 → 实验更干净。

参考文献：
  [1] Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models."
      ICLR 2022.
  [2] Sablayrolles et al. "Radioactive data: tracing through training."
      ICML 2020.
"""
from __future__ import annotations

import os
import gc
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from ..config import RadioactiveConfig


class WatermarkedImageDataset(Dataset):
    """
    水印图像数据集。

    支持两种输入：
      1. 图像目录路径（从磁盘加载）
      2. PIL Image 列表（内存）
    """

    def __init__(
        self,
        images: Optional[List[Any]] = None,
        image_dir: Optional[str] = None,
        image_size: int = 512,
        prompts: Optional[List[str]] = None,
    ):
        self.image_size = image_size
        self.prompts = prompts or []

        if images is not None:
            self.images = images
            self.from_disk = False
        elif image_dir is not None:
            exts = {".png", ".jpg", ".jpeg", ".webp"}
            self.image_paths = sorted([
                os.path.join(image_dir, f) for f in os.listdir(image_dir)
                if os.path.splitext(f)[1].lower() in exts
            ])
            self.images = None
            self.from_disk = True
        else:
            raise ValueError("需要 images 或 image_dir")

    def __len__(self) -> int:
        if self.from_disk:
            return len(self.image_paths)
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.from_disk:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        else:
            img = self.images[idx]
            if not isinstance(img, Image.Image):
                img = Image.fromarray((np.array(img) * 255).astype(np.uint8))

        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        img_tensor = img_tensor * 2.0 - 1.0

        prompt = self.prompts[idx % len(self.prompts)] if self.prompts else ""

        return {"pixel_values": img_tensor, "prompt": prompt}


class LoRAFineTuner:
    """
    LoRA 微调器：验证放射性的核心组件。

    用法：
        finetuner = LoRAFineTuner(config)
        finetuned_pipe = finetuner.finetune(base_pipe, watermarked_images)
        new_images = finetuner.generate(finetuned_pipe, prompts, n=500)
    """

    def __init__(self, config: RadioactiveConfig):
        self.config = config
        self.device = torch.device(config.device)

    def finetune(
        self,
        pipe: Any,
        images: Optional[List[Any]] = None,
        image_dir: Optional[str] = None,
        prompts: Optional[List[str]] = None,
    ) -> Any:
        """
        用水印图像 LoRA 微调 Stable Diffusion UNet。

        这是定理 D 验证的关键步骤：
          微调后的模型 M₁ 应当继承训练数据的拓扑分布，
          因此 M₁ 的输出应当保留水印的拓扑指纹。

        Args:
            pipe:     StableDiffusionPipeline
            images:   水印图像列表
            image_dir: 或水印图像目录
            prompts:  对应提示词

        Returns:
            微调后的 pipeline（带 LoRA 权重）
        """
        from peft import LoraConfig, get_peft_model

        dataset = WatermarkedImageDataset(
            images=images,
            image_dir=image_dir,
            image_size=self.config.image_size,
            prompts=prompts or self.config.prompts,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.lora_batch_size,
            shuffle=True,
            drop_last=True,
        )

        unet = pipe.unet
        vae = pipe.vae
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer
        noise_scheduler = pipe.scheduler

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=0.0,
            bias="none",
        )
        unet = get_peft_model(unet, lora_config)

        trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        total = sum(p.numel() for p in unet.parameters())
        print(f"[LoRA] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        optimizer = torch.optim.AdamW(
            [p for p in unet.parameters() if p.requires_grad],
            lr=self.config.lora_lr,
            weight_decay=1e-4,
        )

        unet.train()
        global_step = 0
        total_steps = self.config.lora_epochs * len(dataloader)

        print(f"[LoRA] Starting fine-tuning: {self.config.lora_epochs} epochs, "
              f"{len(dataloader)} steps/epoch, {total_steps} total steps")

        for epoch in range(self.config.lora_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(self.device, dtype=vae.dtype)
                prompts_batch = batch["prompt"]

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=self.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    text_inputs = tokenizer(
                        list(prompts_batch),
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    encoder_hidden = text_encoder(
                        text_inputs.input_ids.to(self.device)
                    )[0]

                noise_pred = unet(
                    noisy_latents.to(dtype=unet.dtype),
                    timesteps,
                    encoder_hidden_states=encoder_hidden.to(dtype=unet.dtype),
                ).sample

                loss = F.mse_loss(noise_pred.float(), noise.float())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in unet.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if (epoch + 1) % max(1, self.config.lora_epochs // 10) == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{self.config.lora_epochs}  loss={avg_loss:.6f}  "
                      f"step={global_step}/{total_steps}")

        out_dir = os.path.join(self.config.output_dir, "lora_weights")
        os.makedirs(out_dir, exist_ok=True)
        unet.save_pretrained(out_dir)
        print(f"[LoRA] Weights saved to {out_dir}")

        pipe.unet = unet
        return pipe

    def generate(
        self,
        pipe: Any,
        prompts: List[str],
        n_images: int = 500,
        save_dir: Optional[str] = None,
    ) -> List[Any]:
        """
        用微调后的模型生成图像（供放射性检测使用）。

        这些图像应当携带训练数据的拓扑指纹（若放射性成立）。
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        pipe.unet.eval()
        images = []

        print(f"[LoRA] Generating {n_images} images from fine-tuned model...")

        with torch.no_grad():
            for i in range(n_images):
                prompt = prompts[i % len(prompts)]
                result = pipe(
                    prompt,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                )
                image = result.images[0]
                images.append(image)

                if save_dir:
                    image.save(os.path.join(save_dir, f"gen_{i:05d}.png"))

                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{n_images}] generated")

        return images
