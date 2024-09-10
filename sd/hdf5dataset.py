from torch.utils.data import Dataset
from pathlib import Path
import h5py
from transformers import AutoTokenizer
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import torch
import random
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision.transforms.functional import crop
import cv2


def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds



class HDF5Dataset(Dataset):
    """
    Dataset class to iterate over the blenderproc generated hdf5 synthetic dataset for SD 1.5
    TODO: Add support for random masks
    """

    def __init__(
        self,
        data_root: str,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        resolution: int = 512,
        random_mask: bool = False,
        proportion_empty_prompts: float = 0.1,
    ):
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.random_mask = random_mask
        self.data_root = Path(data_root)
        self.df = df
        self.proportion_empty_prompts = proportion_empty_prompts
        self.mirror_prompt = "A perfect plane mirror reflection of "

    def __len__(self):
        return self.df.shape[0]

    def tokenize_caption(self, caption: str):
        if random.random() < self.proportion_empty_prompts:
            caption = ""
        elif isinstance(caption, str):
            caption = self.mirror_prompt + caption
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    @staticmethod
    def get_masked_image(image, mask):
        masked_image = image.copy()
        masked_image[mask == 255] = 0
        return masked_image

    def apply_transforms_rgb(self, image: np.ndarray):
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    self.resolution, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                # transforms.CenterCrop(self.resolution),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        image = image_transforms(image)
        return image

    def apply_transforms_mask(self, mask: np.ndarray):
        mask = torch.tensor(mask, dtype=torch.float32) / 255.0
        mask = mask.unsqueeze(0)
        mask = transforms.Resize(
            self.resolution, interpolation=transforms.InterpolationMode.BICUBIC
        )(mask)
        return mask

    @staticmethod
    def extract_data_from_hdf5(hdf5_data):
        # TODO: extract camera poses, depth maps if required
        data = {}
        data["image"] = np.array(hdf5_data["colors"], dtype=np.uint8)
        mask = np.array(hdf5_data["category_id_segmaps"], dtype=np.uint8)
        data["mask"] = (mask == 1).astype(np.uint8) * 255
        data["masked_image"] = HDF5Dataset.get_masked_image(data["image"], data["mask"])
        return data

    def __getitem__(self, index):
        example = {}
        row = self.df.iloc[index]
        caption = str(row["caption"])
        hdf5_path = self.data_root / str(row["path"])
        hdf5_data = h5py.File(hdf5_path, "r")

        data = self.extract_data_from_hdf5(hdf5_data)

        image = self.apply_transforms_rgb(data["image"])
        mask = self.apply_transforms_mask(data["mask"])
        masked_image = self.apply_transforms_rgb(data["masked_image"])

        example["prompt_ids"] = self.tokenize_caption(caption)
        example["images"] = image
        example["conditioning_images"] = masked_image
        example["masks"] = mask

        return example


if __name__ == "__main__":
        # Load the tokenizers
    from transformers import AutoTokenizer
    from transformers import PretrainedConfig

    def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
    ):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder, revision=revision
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "CLIPTextModelWithProjection":
            from transformers import CLIPTextModelWithProjection

            return CLIPTextModelWithProjection
        else:
            raise ValueError(f"{model_class} is not supported.")


    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    tokenizer_one = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    df = pd.read_csv("/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/data/blenderproc/train.csv")

    train_dataset = HDF5Dataset(
        data_root="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/data/blenderproc",
        df=df,
        tokenizer=tokenizer_one,

    )
    for item in train_dataset:
        print(item)
        break
    
    # train_dataset = InpaintingDataset("/home/vbr/om/BrushNet/data/MSD/train")
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     shuffle=False,
    #     collate_fn=CollateDataset(resolution = 1024, text_encoders = text_encoders, tokenizers = tokenizers),
    #     batch_size=8,
    #     num_workers=1,
    # )
    # # INSERT_YOUR_CODE
    # for batch in train_dataloader:
    #     for key, value in batch.items():
    #         if isinstance(value, torch.Tensor):
    #             print(f"{key}: shape={value.shape}, type={type(value)}")
    #         elif isinstance(value, list):
    #             print(f"{key}: length={len(value)}, type={type(value)}")
    #             if len(value) > 0:
    #                 first_element = value[0]
    #                 print(f"{key} contents: type of first element={type(first_element)}")
    #                 if isinstance(first_element, torch.Tensor):
    #                     print(f"{key} contents: shape of first element={first_element.shape}")
    #         elif isinstance(value, str):
    #             print(f"{key}: value={value}, type={type(value)}")
    #         else:
    #             print(f"{key}: type={type(value)}")
    #     break  # Remove this line if you want to print shapes and types for all batches
