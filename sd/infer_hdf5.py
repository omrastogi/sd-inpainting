import autoroot  # root setup, do not delete
import autorootcwd  # root setup, do not delete
import os
import torch
from PIL import Image, ImageFilter
from diffusers import (
    StableDiffusionInpaintPipeline, 
    UNet2DConditionModel,
    DDPMScheduler
)
from transformers import CLIPTextModel
import pandas as pd 
from hdf5dataset import HDF5Dataset
import h5py
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json  # Added missing dependency

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from metrics.metrics import compute_metrics

def init_pipeline(model_path, pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting"):
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.float32,
        revision=None
    )

    pipeline.unet = UNet2DConditionModel.from_pretrained(
        model_path, subfolder="unet", revision=None,
    )
    pipeline.text_encoder = CLIPTextModel.from_pretrained(
        model_path, subfolder="text_encoder", revision=None,
    )
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to("cuda")

    return pipeline 


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Stable Diffusion Inpainting")
    parser.add_argument(
        "--pretrained_model_name_or_path",type=str, default=os.environ["MODEL_NAME"], required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--model_path", type=str, 
                        default="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/Reflection-Exploration/BrushNet/baseline/sd_inpainting/model_lora_abo", 
                        help="Path to the model")
    parser.add_argument("--data_dir", type=str, 
                        default="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/data/blenderproc", 
                        help="Path to the training data directory")
    parser.add_argument("--eval_dir", type=str, 
                        default="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/runs/sd_inpainting", 
                        help="Path to the training data directory")
    parser.add_argument("--test_csv", type=str, default="test.csv", help="The CSV file containing the test data.")
    parser.add_argument("--num_validation_images", type=int, default=4, help="Number of validation images")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    return parser.parse_args()

def inference(df, args, device):
    def get_data(index):
        row = df.iloc[index]
        caption = str(row["caption"])
        hdf5_path = os.path.join(args.data_dir, str(row["path"]))
        hdf5_data = h5py.File(hdf5_path, "r")
        return hdf5_data, caption, hdf5_path
    
    pipeline = init_pipeline(args.model_path, args.pretrained_model_name_or_path)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # assert df.shape[0] == args.num_validation_images, "Number of validation images should be equal to the number of images in the test dataset"
    print(df.shape)    
    image_logs = []
    mirror_prompt = "A perfect plane mirror reflection of "
    all_metrics = {
        "psnr": [],
        "ssim": [],
        "lpips": []
    }

    for i in range(len(df)):
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

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(os.path.join(args.data_dir, args.test_csv))
    
    os.makedirs(args.eval_dir, exist_ok=True)
    os.makedirs(os.path.join(args.eval_dir, "output"), exist_ok=True)
    
    for dataset_metric, image_log in inference(df, args, device="cuda"):
        # Save dataset metric
        metric_path = os.path.join(args.eval_dir, "dataset_metric.json")
        with open(metric_path, "w") as f:
            json.dump(dataset_metric, f)
        
        # Save images as grid
        images = image_log["images"]
        grid = image_grid(images, rows=1, cols=len(images))
        grid_path = os.path.join(args.eval_dir, "output", f"{image_log['filepath'].split('/')[-2]}_{image_log['filepath'].split('/')[-1].split('.')[0]}.png")
        grid.save(grid_path)