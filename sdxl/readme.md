## Setup
```bash
pip install -r requirements.txt
pip install autoroot autorootcwd pandas wandb bitsandbytes
```
## How to run
```bash
CUDA_VISIBLE_DEVICES=1 python baseline/sdxl_inpainting/train_sdxl_hdf5.py \
--pretrained_model_name_or_path="diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
--train_data_dir="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/data/blenderproc" \
--output_dir="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/run/sdxl_all" \
--resolution=1024 \
--train_batch_size=8 \
--gradient_accumulation_steps=1 --gradient_checkpointing \
--use_8bit_adam \
--set_grads_to_none \
--unet_learning_rate=1e-4 \
--text_encoder_learning_rate=1e-5 \
--lr_scheduler="constant" \
--lr_warmup_steps=100 \
--max_train_steps=5 \
--lora_rank=12 \
--lora_dropout=0.3 \
--checkpointing_steps=1 \
--lora_alpha=20
```

### infer_hdf5_2
```bash
CUDA_VISIBLE_DEVICES=1 python baseline/sdxl_inpainting/sdxl_infer_hdf5.py \
--pretrained_model_name_or_path="diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
--data_dir="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/data/blenderproc" \
--model_path="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/run/sdxl_all" \
--lora_rank=12 \
--lora_dropout=0.3 \
--lora_alpha=20
```