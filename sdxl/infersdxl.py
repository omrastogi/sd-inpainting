from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import os
import glob
import pandas as pd

def is_image(path):
    ext = os.path.splitext(path.lower())[-1]
    return ext == ".png" or ext == ".jpg"

class InpaintingPipeline:
    def __init__(self, model_path, img_dir, mask_dir, prompt_csv, seed, output_dir):
        self.pipe = AutoPipelineForInpainting.from_pretrained(model_path, torch_dtype=torch.float16, variant="fp16").to("cuda")
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.prompt_csv = prompt_csv
        self.seed = seed
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def process_images(self):
        img_files = sorted(filter(is_image, glob.glob("{}/*".format(self.img_dir))))
        mask_files = sorted(filter(is_image, glob.glob("{}/*".format(self.mask_dir))))
        prompt_list = pd.read_csv(self.prompt_csv)

        for img_path, mask_path in zip(img_files, mask_files):
            image = load_image(img_path).resize((1024, 1024))
            mask_image = load_image(mask_path).resize((1024, 1024))
            img_filename = os.path.basename(img_path).split('.')[0] + ".jpg"
            prompt = prompt_list[prompt_list["images"] == img_filename]['Captions'].iloc[0]
            print(img_filename)
            image = self.pipe(
                prompt=prompt,
                image=image,
                mask_image=mask_image,
                guidance_scale=8.0,
                num_inference_steps=20,  # steps between 15 and 30 work well for us
                strength=0.99,  # make sure to use `strength` below 1.0
                generator=torch.Generator(device="cuda").manual_seed(self.seed),
            ).images[0]

            image.save(os.path.join(self.output_dir, f"{os.path.basename(img_path)}"))

inpainting_pipeline = InpaintingPipeline(
    model_path="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    img_dir="dataset/MSD/test/images",
    mask_dir="dataset/MSD/test/masks",
    prompt_csv="captions.csv",
    seed=0,
    output_dir="generated_images_test"
)
inpainting_pipeline.process_images()
