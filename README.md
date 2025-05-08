# img2img-turbo


- The following command takes a **rainy** image file as input, and saves the output **clear** in the directory specified.
    ```bash
    python src/inference_unpaired.py --model_name "rainy_to_clear" \
        --input_image data/cloudy/sequences/val/frames/sequence924/sequence924_frame154224_info.png --output_dir "outputs"

    python src/inference_unpaired.py --model_name "clear_to_rainy" --input_image data/sunny/sequences/val/frames/sequence99/sequence636_frame96312_info.png --output_dir "outputs"
    
    ```

