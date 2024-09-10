
## Setup
```bash
pip install -r requirements.txt
pip install autoroot autorootcwd pandas wandb bitsandbytes
```

## Env
Create .env 
```bash
CUDA_VISIBLE_DEVICES="0"
MODEL_NAME="diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
WANDB_API_KEY=""
WANDB_ENTITY="omegam"
WANDB_PROJECT_NAME="sd_inpainting"
TRAIN_DIR="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/data/blenderproc"
OUTPUT_DIR="model"
```

## Training sd inpainting lora
```bash
CUDA_VISIBLE_DEVICES=0 python baseline/sd_inpainting/train_sdinpainting_lora.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting" \
--train_data_dir="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/data/blenderproc" \
--validation_steps=500 \
--output_dir="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/run/sdinpainting_abo" \
--resolution=1024 \
--train_batch_size=8 \
--gradient_accumulation_steps=1 --gradient_checkpointing \
--use_8bit_adam \
--set_grads_to_none \
--unet_learning_rate=1e-4 \
--lr_scheduler="constant" \
--lr_warmup_steps=100 \
--max_train_steps=12000 \
--lora_rank=12 \
--lora_dropout=0.3 \
--checkpointing_steps=1000 \
--lora_alpha=20
```


## Evaluating sd inpainting lora

### sd_infer_hdf5
```bash
CUDA_VISIBLE_DEVICES=0 python baseline/sd_inpainting/sd_infer_hdf5.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting" \
--train_data_dir="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/data/blenderproc" \
--model_path="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/run/sdinpainting_abo" \
--lora_rank=12 \
--lora_dropout=0.3 \
--lora_alpha=20
```

### infer_hdf5 - Don't use this code
```bash
CUDA_VISIBLE_DEVICES=1 python baseline/sd_inpainting/infer_hdf5.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting" \
--model_path="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/run/sdinpainting_abo" \
--data_dir="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/data/blenderproc" \
--eval_dir="output" \
--test_csv="test.csv" \
--num_validation_images=4 \
--num_inference_steps=50 \
--seed=42
```

## Training sd inpainting 
The result are not comparabe and code is deprecated. Also no inference code provided. 

```bash
CUDA_VISIBLE_DEVICES=0 python baseline/sd_inpainting/train_sdinpainting.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting" \
--train_data_dir="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/data/blenderproc" \
--train_csv="train_abo.csv" \
--test_csv="test_abo.csv" \
--validation_steps=2 \
--output_dir="model" \
--resolution=1024 \
--train_batch_size=8 \
--gradient_accumulation_steps=1 --gradient_checkpointing \
--use_8bit_adam \
--set_grads_to_none \
--unet_learning_rate=1e-4 \
--text_encoder_learning_rate=1e-5 \
--lr_scheduler="constant" \
--lr_warmup_steps=100 \
--max_train_steps=20000 \
--checkpointing_steps=4000
```

