import os
import argparse
import random

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def compute_sunny_metrics(img: Image.Image) -> dict:
    """
    Computes extended heuristic metrics for 'sunny-ness':
      - Brightness (mean grayscale)
      - Saturation (mean S channel in HSV)
      - Contrast (std dev of grayscale)
      - Sky-blue ratio (fraction of pixels likely sky: blue hue, mid/high saturation, brightness)
      - Highlight ratio (fraction of very bright pixels)
      - Dark channel prior (mean of min RGB channel, indicates haze/shadow)
    """
    rgb = np.array(img).astype(np.float32) / 255.0
    gray = np.array(img.convert('L')).astype(np.float32) / 255.0
    hsv = np.array(img.convert('HSV')).astype(np.float32) / 255.0

    brightness = gray.mean()
    contrast = gray.std()
    saturation = hsv[:, :, 1].mean()

    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sky_mask = (h >= 0.25) & (h <= 0.45) & (s > 0.2) & (v > 0.3)
    sky_ratio = sky_mask.mean()

    highlight_ratio = (gray > 0.9).mean()
    dark_channel = np.min(rgb, axis=2).mean()

    return {
        'brightness': brightness,
        'saturation': saturation,
        'contrast': contrast,
        'sky_ratio': sky_ratio,
        'highlight_ratio': highlight_ratio,
        'dark_channel': dark_channel
    }

def average_metrics_in_directory(directory: str) -> dict:
    """
    Computes average metrics over all images in a directory.
    """
    accum = None
    count = 0

    for fname in os.listdir(directory):
        path = os.path.join(directory, fname)
        try:
            img = Image.open(path).convert('RGB')
        except:
            continue
        metrics = compute_sunny_metrics(img)
        if accum is None:
            accum = metrics.copy()
        else:
            for k, v in metrics.items():
                accum[k] += v
        count += 1

    if count == 0:
        raise ValueError(f"No images found in directory {directory}")

    return {k: accum[k] / count for k in accum}

def process_videos(args, mode: str):
    """
    Process videos for either 'sunny' or 'cloudy' mode, based on args.
    Extracts frames, filters by metrics, resizes, and splits into train/val.
    Shows progress with tqdm.
    """
    if mode == 'sunny':
        src_dir = args.sunny_videos
        dst_dir = args.sunny_dst
        samples_dir = args.sunny_samples
    else:
        src_dir = args.cloudy_videos
        dst_dir = args.cloudy_dst
        samples_dir = args.cloudy_samples

    avg_metrics = average_metrics_in_directory(samples_dir)
    print(f"Average metrics for {mode} samples:", avg_metrics)

    train_dir = os.path.join(dst_dir, 'train')
    val_dir   = os.path.join(dst_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for video_fname in os.listdir(src_dir):
        video_path = os.path.join(src_dir, video_fname)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[{mode}] Failed to open {video_fname}")
            continue

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if video_fps <= 0 or total_frames <= 0:
            cap.release()
            print(f"[{mode}] Invalid FPS or frame count for {video_fname}")
            continue

        duration = total_frames / video_fps
        skip_secs = 60.0 if args.skip_first_minute else 0.0
        effective_duration = max(0.0, duration - skip_secs)
        expected_frames = int(effective_duration * args.fps)
        print(f"[{mode}] {video_fname}: duration {duration:.1f}s, "
              f"expected ~{expected_frames} frames at {args.fps}fps"
              f"{' (skipping first minute)' if args.skip_first_minute else ''}")

        frame_interval = max(1, int(round(video_fps / args.fps)))
        skip_frames = int(video_fps * skip_secs)

        basename, _ = os.path.splitext(video_fname)
        frame_idx = 0
        saved_count = 0

        # Progress bar over total frames
        with tqdm(total=int(total_frames), desc=f"{mode} {video_fname}", ncols=100) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                pbar.update(1)

                if frame_idx < skip_frames:
                    frame_idx += 1
                    continue

                if frame_idx % frame_interval != 0:
                    frame_idx += 1
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)

                metrics = compute_sunny_metrics(img)
                if mode == 'sunny':
                    below = sum(1 for k, v in metrics.items() if v < avg_metrics[k])
                    discard = below > 3
                else:
                    above = sum(1 for k, v in metrics.items() if v > avg_metrics[k])
                    discard = above > 3

                if not discard:
                    w, h = img.size
                    max_side = max(w, h)
                    if max_side > args.max_resolution:
                        scale = args.max_resolution / max_side
                        new_size = (int(w * scale), int(h * scale))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)

                    subset = 'val' if random.random() < args.split else 'train'
                    out_dir = train_dir if subset == 'train' else val_dir

                    out_fname = f"{basename}_frame{frame_idx:06d}.jpg"
                    img.save(os.path.join(out_dir, out_fname))
                    saved_count += 1
                else:
                    print(f"[{mode}] Discarded frame {frame_idx} for {video_fname}")

                frame_idx += 1

        cap.release()
        print(f"[{mode}] Processed {video_fname}: saved {saved_count} frames")

def main():
    parser = argparse.ArgumentParser(
        description="Extract and filter frames from sunny/cloudy videos"
    )
    parser.add_argument(
        "--sunny_videos", default="data/raw/sunny",
        help="Directory containing sunny videos"
    )
    parser.add_argument(
        "--cloudy_videos", default="data/raw/cloudy",
        help="Directory containing cloudy videos"
    )
    parser.add_argument(
        "--sunny_dst", default="data/target",
        help="Output directory for filtered sunny frames"
    )
    parser.add_argument(
        "--cloudy_dst", default="data/source",
        help="Output directory for filtered cloudy frames"
    )
    parser.add_argument(
        "--sunny_samples", default="samples/sunny",
        help="Directory containing sample sunny images"
    )
    parser.add_argument(
        "--cloudy_samples", default="samples/cloudy",
        help="Directory containing sample cloudy images"
    )
    parser.add_argument(
        "--fps", type=int, required=True,
        help="Frames per second to extract"
    )
    parser.add_argument(
        "--split", type=float, default=0.2,
        help="Fraction of frames to allocate to validation (val), rest to training"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split"
    )
    parser.add_argument(
        "--skip_first_minute", action="store_true",
        help="Skip extracting frames from the first minute of each video"
    )
    parser.add_argument(
        "--max_resolution", type=int, default=1080,
        help="Maximum size for the largest side of exported frames"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    process_videos(args, 'sunny')
    process_videos(args, 'cloudy')

if __name__ == "__main__":
    main()

