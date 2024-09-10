import autoroot  # root setup, do not delete
import autorootcwd  # root setup, do not delete
import random
import argparse
import copy
import itertools
import logging
import math
import os
import gc
import shutil
import h5py
from pathlib import Path
import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms_v2
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from hdf5dataset import HDF5Dataset
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from peft import PeftModel, LoraConfig, get_peft_model
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json  # Added missing dependency

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from metrics.metrics import compute_metrics

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=os.environ["MODEL_NAME"],
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=os.environ["TRAIN_DIR"],
        required=True,
        help="A folder containing the training data of images.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/Reflection-Exploration/BrushNet/baseline/sd_inpainting/model_lora_abo",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--num_validation_images", type=int, default=4, help="Number of validation images")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='latest',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=27,
        help=("The alpha constant of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="The dropout rate of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        help="The bias type of the Lora update matrices. Must be 'none', 'all' or 'lora_only'.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def inference(df, args):
    def get_data(index):
        row = df.iloc[index]
        caption = str(row["caption"])
        hdf5_path = os.path.join(args.train_data_dir, str(row["path"]))
        hdf5_data = h5py.File(hdf5_path, "r")
        return hdf5_data, caption, hdf5_path

    pipeline, accelerator = create_pipeline(args)
    print("Pipeline Created")

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # assert df.shape[0] == args.num_validation_images, "Number of validation images should be equal to the number of images in the test dataset"
    image_logs = []
    mirror_prompt = "A perfect plane mirror reflection of "
    all_metrics = {
        "psnr": [],
        "ssim": [],
        "lpips": []
    }

    for i in tqdm(range(len(df)), desc="Processing images"):
        hdf5_data, caption, hdf5_path = get_data(i)
        validation_prompt = mirror_prompt + caption
        data = HDF5Dataset.extract_data_from_hdf5(hdf5_data)
        validation_full_image = Image.fromarray(data["image"], mode="RGB")
        validation_image = Image.fromarray(data["masked_image"], mode="RGB")
        validation_mask = Image.fromarray(data["mask"]).convert("RGB")

        images = []
        metrics = {
            "psnr": [],
            "ssim": [],
            "lpips": []
        }

        for _ in range(args.num_validation_images):
            image = pipeline(
                prompt=validation_prompt,
                image=validation_image,
                mask_image=validation_mask,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=1,
                generator=generator,
            ).images[0]


            metric = compute_metrics(np.array(image), np.array(validation_full_image))
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            text = f"PSNR: {metric['psnr']:.2f}\nSSIM: {metric['ssim']:.2f}\nLPIPS: {metric['lpips']:.2f}"
            draw.text((10, 10), text, font=font, fill=(0, 255, 0))  # Using green color to highlight the text
            images.append(image)
            metrics["psnr"].append(metric["psnr"].item())
            metrics["ssim"].append(metric["ssim"].item())
            metrics["lpips"].append(metric["lpips"].item())

        all_metrics["psnr"].append(max(metrics["psnr"]))
        all_metrics["ssim"].append(max(metrics["ssim"]))
        all_metrics["lpips"].append(min(metrics["lpips"]))

    
        image_log = {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt, "filepath": hdf5_path}
        image_logs.append(image_log)

        dataset_metric = {
        "psnr": np.mean(all_metrics["psnr"]),
        "ssim": np.mean(all_metrics["ssim"]),
        "lpips": np.mean(all_metrics["lpips"])
        } 

        yield dataset_metric, image_log


def create_pipeline(args):

    accelerator = Accelerator()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            use_fast=False,
        )

    # Load scheduler and models
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", 
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", 
    )

    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
    )
    unet = get_peft_model(unet, config)

    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["k_proj", "q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
    )
    text_encoder = get_peft_model(text_encoder, config)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                sub_dir = "unet" if isinstance(model.base_model.model, type(accelerator.unwrap_model(unet).base_model.model)) else "text_encoder"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            sub_dir = "unet" if isinstance(model.base_model.model, type(accelerator.unwrap_model(unet).base_model.model)) else "text_encoder"
            model_cls = UNet2DConditionModel if isinstance(model.base_model.model, type(accelerator.unwrap_model(unet).base_model.model)) else CLIPTextModel

            load_model = model_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder=sub_dir)
            load_model = PeftModel.from_pretrained(load_model, input_dir, subfolder=sub_dir)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    # Prepare everything with our `accelerator`.
    unet, text_encoder = accelerator.prepare(unet, text_encoder)

    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    # Move vae to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.model_path)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None

        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.model_path, path))
            
    else:
        pass

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        safety_checker=None,
    )

    # set `keep_fp32_wrapper` to True because we do not want to remove
    # mixed precision hooks while we are still training
    pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
    pipeline.text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    return pipeline, accelerator
    
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(os.path.join(args.train_data_dir, "test.csv"))

    eval_dir="output"
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(os.path.join(eval_dir, "output"), exist_ok=True)
    
    for dataset_metric, image_log in inference(df, args):
        # Save dataset metric
        metric_path = os.path.join(eval_dir, "dataset_metric.json")
        with open(metric_path, "w") as f:
            json.dump(dataset_metric, f)
        
        # Save images as grid
        images = image_log["images"]
        grid = image_grid(images, rows=1, cols=len(images))
        grid_path = os.path.join(eval_dir, "output", f"{image_log['filepath'].split('/')[-2]}_{image_log['filepath'].split('/')[-1].split('.')[0]}.png")
        grid.save(grid_path)
