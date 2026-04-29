import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--class_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--instance_prompt", type=str, required=True)
    parser.add_argument("--class_prompt", type=str, required=True)
    parser.add_argument("--num_class_images", type=int, default=1000)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_dir, class_data_dir, instance_prompt, class_prompt, tokenizer, resolution=512):
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt

        self.instance_images = list(Path(instance_data_dir).iterdir())
        self.class_images = list(Path(class_data_dir).iterdir())
        self.num_instance_images = len(self.instance_images)
        self.num_class_images = len(self.class_images)
        self._length = max(self.num_class_images, self.num_instance_images)

        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        instance_image = Image.open(self.instance_images[index % self.num_instance_images]).convert("RGB")
        instance_image = self.image_transforms(instance_image)
        instance_tokens = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        class_image = Image.open(self.class_images[index % self.num_class_images]).convert("RGB")
        class_image = self.image_transforms(class_image)
        class_tokens = self.tokenizer(
            self.class_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "instance_images": instance_image,
            "instance_prompt_ids": instance_tokens,
            "class_images": class_image,
            "class_prompt_ids": class_tokens,
        }


def generate_class_images(args, pipeline, device):
    class_data_dir = Path(args.class_data_dir)
    class_data_dir.mkdir(parents=True, exist_ok=True)
    existing = list(class_data_dir.iterdir())

    if len(existing) >= args.num_class_images:
        return

    num_to_generate = args.num_class_images - len(existing)
    print(f"Generating {num_to_generate} class prior images...")

    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    batch_size = 4
    for i in tqdm(range(0, num_to_generate, batch_size)):
        current_batch = min(batch_size, num_to_generate - i)
        images = pipeline(
            [args.class_prompt] * current_batch,
            num_inference_steps=50,
            guidance_scale=7.5,
        ).images
        for j, img in enumerate(images):
            img.save(class_data_dir / f"class_{len(existing) + i + j:04d}.png")

    pipeline.to("cpu")
    del pipeline
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    device = get_device()
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if device.type == "mps":
        weight_dtype = torch.float32

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder", torch_dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae", torch_dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet", torch_dtype=weight_dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    generate_class_images(args, pipeline, device)

    vae.requires_grad_(False)
    vae.to(device)
    text_encoder.requires_grad_(False)
    text_encoder.to(device)
    text_encoder.eval()
    unet.to(device)
    unet.train()

    optimizer_cls = bnb.optim.AdamW8bit if HAS_BNB else torch.optim.AdamW
    optimizer = optimizer_cls(unet.parameters(), lr=args.learning_rate)

    dataset = DreamBoothDataset(
        instance_data_dir=args.instance_data_dir,
        class_data_dir=args.class_data_dir,
        instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        resolution=args.resolution,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    use_autocast = args.mixed_precision != "no" and device.type == "cuda"
    # GradScaler requires fp32 parameter gradients; when models are loaded in fp16 the
    # gradients are already fp16 and cannot be unscaled, so we disable the scaler.
    use_scaler = use_autocast and weight_dtype == torch.float32
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    progress_bar = tqdm(range(args.max_train_steps), desc="Training")
    global_step = 0

    while global_step < args.max_train_steps:
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break

            with torch.amp.autocast(device_type=device.type, enabled=use_autocast):
                instance_images = batch["instance_images"].to(device, dtype=weight_dtype)
                instance_prompt_ids = batch["instance_prompt_ids"].to(device)
                class_images = batch["class_images"].to(device, dtype=weight_dtype)
                class_prompt_ids = batch["class_prompt_ids"].to(device)

                latents_instance = vae.encode(instance_images).latent_dist.sample() * vae.config.scaling_factor
                latents_class = vae.encode(class_images).latent_dist.sample() * vae.config.scaling_factor

                noise_instance = torch.randn_like(latents_instance)
                noise_class = torch.randn_like(latents_class)

                timesteps_instance = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents_instance.shape[0],), device=device).long()
                timesteps_class = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents_class.shape[0],), device=device).long()

                noisy_latents_instance = noise_scheduler.add_noise(latents_instance, noise_instance, timesteps_instance)
                noisy_latents_class = noise_scheduler.add_noise(latents_class, noise_class, timesteps_class)

                encoder_hidden_states_instance = text_encoder(instance_prompt_ids)[0]
                encoder_hidden_states_class = text_encoder(class_prompt_ids)[0]

                noise_pred_instance = unet(noisy_latents_instance, timesteps_instance, encoder_hidden_states_instance).sample
                noise_pred_class = unet(noisy_latents_class, timesteps_class, encoder_hidden_states_class).sample

                loss_instance = F.mse_loss(noise_pred_instance, noise_instance, reduction="mean")
                loss_class = F.mse_loss(noise_pred_class, noise_class, reduction="mean")

                loss = loss_instance + args.prior_loss_weight * loss_class

            scaler.scale(loss).backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item(), loss_inst=loss_instance.item(), loss_cls=loss_class.item())
            global_step += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        unet=unet,
        text_encoder=text_encoder,
        safety_checker=None,
    )
    pipeline.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
