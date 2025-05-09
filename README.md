# img2img-turbo

# train

```bash
accelerate config

export NCCL_P2P_DISABLE=1
accelerate launch --main_process_port 29501 src/train_cyclegan_turbo.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="output/cyclegan_turbo/cloudy2sunny" \
    --dataset_folder "data/cloudy2sunny" \
    --train_img_prep "resize_286_randomcrop_256x256_hflip" --val_img_prep "no_resize" \
    --learning_rate="1e-5" --max_train_steps=25000 \
    --train_batch_size=1 --gradient_accumulation_steps=1 \
    --report_to "wandb" --tracker_project_name "gparmar_unpaired_h2z_cycle_debug_v2" \
    --enable_xformers_memory_efficient_attention --validation_steps 250 \
    --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1
```


- The following command takes a **rainy** image file as input, and saves the output **clear** in the directory specified.
    ```bash
    python src/inference_unpaired.py --model_name "rainy_to_clear" \
        --input_image data/cloudy/sequences/val/frames/sequence924/sequence924_frame154224_info.png --output_dir "outputs"

    python src/inference_unpaired.py --model_name "clear_to_rainy" --input_image data/sunny/sequences/val/frames/sequence99/sequence636_frame96312_info.png --output_dir "outputs"
    
    ```

