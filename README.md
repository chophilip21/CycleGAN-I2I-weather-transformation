# VSAIT: Unpaired Image Translation via Vector Symbolic Architectures

Install dependencies via pip:
```bash
pip install -r requirements.txt
```

### Dataset Preparation
For any two image datasets with png/jpg images, download source and target data (or create symlinks) to `./data/source/` and `./data/target/` with `train` and `val` subfolders for each domain.

The dataset has been extracted from self collected videos and the `extractor.py` script. Adjust the parameters as needed. It uses some heuristics to filter out images that fall below a threshold for brightness, saturation, contrast, sky ratio, highlight ratio, and dark channel prior. The source data will not be disclosed due to privacy concerns. You can also use [INIT dataset](https://zhiqiangshen.com/projects/INIT/index.html) as a starting point. 


### Training
Launch training with defaults in configs:
```bash
# 200000 steps
python train.py --name="vsait"
```

This will use the default configs in `./configs/` and save checkpoints and translated images in `./checkpoints/vsait/`.

### Evaluation
Translate images in `./data/source/val/` using a specific checkpoint:
```bash
python test.py --name="vsait_adapt" --checkpoint="./checkpoints/vsait/version_0/checkpoints/epoch={i}-step={j}.ckpt"
```

Images from the above example would be saved in `./checkpoints/vsait_adapt/images/`.

