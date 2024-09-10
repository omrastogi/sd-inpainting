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

class RandomCrop:
    def __init__(self, image, crop_size):
        """
        Initialize the RandomCrop class with the image and crop size.

        Args:
            image (np.array): The input image as a NumPy array.
            crop_size (tuple): The desired crop size (height, width).
        """
        self.image = image
        self.crop_size = crop_size
        self.crop_params = self._get_crop_params()

    def _get_crop_params(self):
        """
        Get the crop parameters for the image.

        Returns:
            crop_params (tuple): The crop parameters (y1, x1, h, w).
        """
        img_height, img_width, _ = self.image.shape
        crop_height, crop_width = self.crop_size

        if crop_height > img_height or crop_width > img_width:
            raise ValueError("Crop size must be smaller than the image size.")

        y1 = random.randint(0, img_height - crop_height)
        x1 = random.randint(0, img_width - crop_width)

        return (y1, x1, crop_height, crop_width)

    def crop(self, image):
        """
        Crop the given image using the stored crop parameters.

        Args:
            image (np.array): The input image as a NumPy array.

        Returns:
            cropped_image (np.array): The cropped image.
        """
        y1, x1, crop_height, crop_width = self.crop_params
        cropped_image = image[y1:y1 + crop_height, x1:x1 + crop_width]
        return cropped_image

    def get_crop_params(self):
        """
        Get the crop parameters.

        Returns:
            crop_params (tuple): The crop parameters (y1, x1, h, w).
        """
        return self.crop_params

class RandomResize:
    def __init__(self, min_size, max_size):
        """
        Initialize the RandomResize class with the specified range.

        Args:
            min_size (int): The minimum size for the resize.
            max_size (int): The maximum size for the resize.
        """
        self.new_size = random.randint(min_size, max_size)

    def __call__(self, image):
        """
        Randomly resize a NumPy array image within the specified range.

        Args:
            image (np.array): The input image as a NumPy array.

        Returns:
            resized_image (np.array): The resized image.
        """
        img_height, img_width = image.shape[:2]

        # Randomly choose a new size within the specified range
        
        scaling_factor = self.new_size / min(img_height, img_width)
        new_height = int(img_height * scaling_factor)
        new_width = int(img_width * scaling_factor)

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return resized_image


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
        xl=False
    ):
        self.xl=xl
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

        # normalization_transform = transforms.Compose(
        #     [
        #         transforms.Normalize([0.5], [0.5]),
        #     ]
        # )
        # if self.xl:
        #     image = normalization_transform(image)
        # else:
        #     image = image_transforms(image)
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
    
    def preprocess_for_xl(self, image, mask, masked_image):
        height, width = image.shape[1], image.shape[2]
        random_resize = RandomResize(self.resolution, int(1.125 * self.resolution))
        image, mask, masked_image = random_resize(image), random_resize(mask), random_resize(masked_image)
        random_crop = RandomCrop(image, (self.resolution, self.resolution))
        x1,y1,h,w = random_crop.get_crop_params()
        image, mask, masked_image = random_crop.crop(image), random_crop.crop(mask), random_crop.crop(masked_image)   
        return image, mask, masked_image, (y1, x1), (height, width)
    
    def __getitem__(self, index):
        example = {}
        row = self.df.iloc[index]
        caption = str(row["caption"])
        hdf5_path = self.data_root / str(row["path"])
        hdf5_data = h5py.File(hdf5_path, "r")

        data = self.extract_data_from_hdf5(hdf5_data)

        if self.xl:
            data["image"], data["mask"], data["masked_image"], example["crop_top_left"], example["original_size"] = self.preprocess_for_xl(data["image"], data["mask"], data["masked_image"])
        
        image = self.apply_transforms_rgb(data["image"])
        mask = self.apply_transforms_mask(data["mask"])
        masked_image = self.apply_transforms_rgb(data["masked_image"])

        if self.xl:
            example["prompt_ids"] = caption
        else:
            example["prompt_ids"] = self.tokenize_caption(caption)[0]

        example["pixel_values"] = image
        example["conditioning_pixel_values"] = masked_image
        example["masks"] = mask
        # print(data["image"].shape, data["mask"].shape, data["masked_image"].shape)
        # print(image.shape, mask.shape, masked_image.shape)
        return example

def collate_fn(examples):
    input_ids = [example["prompt_ids"] for example in examples]
    images = [example["pixel_values"] for example in examples]

    masks = [example["masks"] for example in examples]
    conditioning_images = [example["conditioning_pixel_values"] for example in examples]
    
    images = torch.stack(images)
    images = images.to(memory_format=torch.contiguous_format).float()

    masks = torch.stack(masks)
    masks = masks.to(memory_format=torch.contiguous_format).float()

    conditioning_images = torch.stack(conditioning_images)
    conditioning_images = conditioning_images.to(memory_format=torch.contiguous_format).float()

    
    # input_ids = torch.cat(input_ids, dim=0)
    batch = {
        "prompt": input_ids,
        "images": images,
        "masks": masks,
        "conditioning_images": conditioning_images,
    }

    if "crop_top_left" in examples[0]:
        batch["crop_top_left"] = [example["crop_top_left"] for example in examples]
        batch["original_size"] = [example["original_size"] for example in examples]

    return batch


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
        resolution=1024,
        tokenizer=tokenizer_one,
        xl=True

    )
    for i, item in enumerate(train_dataset):
        if i == 10:
            break
    print(train_dataset[0]["pixel_values"].shape)
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
