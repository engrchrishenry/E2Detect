import argparse
import os
import shutil
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This scripts generates timestamp.txt file for Reds 120fps dataset and can resize frames as well. These timestamp.txt files are required for generating events via ESIM.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the REDS 120fps dataset')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Path to the output directory')
    parser.add_argument('--fps', type=float, required=True,
                        help='FPS for input dataset. fps=120 for REDS 120fps dataset.')
    parser.add_argument('--resize', action='store_true',
                        help='Resize frames')
    parser.add_argument('--img_size', type=str,
                        help="Target resolution (e.g., '320:180'). REDS 120fps dataset is 720p 16:9 so choose the target size accordingly.")

    args = parser.parse_args()

    if args.resize and args.img_size is None:
        parser.error("--img_size is required when --resize is specified.")

    target_size = tuple(map(int, args.img_size.split(':')))
    delta_t = 1 / args.fps

    c = 0
    for seq_folder in sorted(os.listdir(args.data_dir)):
        frames = sorted([f for f in os.listdir(f'{args.data_dir}/{seq_folder}') if f.lower().endswith(('.png', '.jpg'))])
        temp_path = os.path.join(args.out_dir, seq_folder)
        os.makedirs(f'{temp_path}/imgs', exist_ok=True)
        for frame in frames:
            if args.resize:
                with Image.open(f'{args.data_dir}/{seq_folder}/{frame}') as img:
                    img_resized = img.resize(target_size)
                    img_resized.save(f'{temp_path}/imgs/{frame}')
            else:
                shutil.copy(f'{args.data_dir}/{seq_folder}/{frame}', f'{temp_path}/imgs')
        timestamp_file = os.path.join(temp_path, f"timestamps.txt")
        with open(timestamp_file, "w") as f:
            for i in range(len(frames)):
                timestamp = round(i * delta_t, 6)
                f.write(f"{timestamp:.6f}\n")
        c += 1
        print(f"Processed folder: {seq_folder} ({c}/{len(os.listdir(args.data_dir))})")

