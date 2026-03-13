import argparse
import os
import shutil
import multiprocessing
from PIL import Image
from joblib import Parallel, delayed


def process_frame(frame_path, output_path, resize, target_size):
    if resize:
        with Image.open(frame_path) as img:
            img_resized = img.resize(target_size, resample=Image.BILINEAR)
            img_resized.save(output_path)
    else:
        shutil.copy(frame_path, output_path)


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
                        help="Target resolution (e.g., '533:300'). REDS 120fps dataset is 720p 16:9 so choose the target size accordingly.")
    parser.add_argument("--cores", type=int, default=-1,
                        help="Number of cores to use to process the data. Default: -1 -> Uses all cores.")

    args = parser.parse_args()
    if args.resize and args.img_size is None:
        parser.error("--img_size is required when --resize is specified.")
    
    target_size = tuple(map(int, args.img_size.split(':')))
    cores = multiprocessing.cpu_count() if args.cores == -1 else args.cores
    delta_t = 1 / args.fps


c = 0
for seq_folder in sorted(os.listdir(args.data_dir)):
    frame_folder = os.path.join(args.data_dir, seq_folder)
    frames = sorted([f for f in os.listdir(frame_folder) if f.lower().endswith(('.png', '.jpg'))])
    temp_path = os.path.join(args.out_dir, seq_folder)
    os.makedirs(os.path.join(temp_path, "imgs"), exist_ok=True)

    # Build job list
    jobs = [
        (os.path.join(frame_folder, frame),
         os.path.join(temp_path, "imgs", frame),
         args.resize,
         target_size)
        for frame in frames
    ]

    # Parallel execution using joblib
    Parallel(n_jobs=cores)(
        delayed(process_frame)(*job) for job in jobs
    )

    # Write timestamps
    timestamp_file = os.path.join(temp_path, "timestamps.txt")
    with open(timestamp_file, "w") as f:
        for i in range(len(frames)):
            f.write(f"{round(i * delta_t, 6):.6f}\n")

    c += 1
    print(f"Processed folder: {seq_folder} ({c}/{len(os.listdir(args.data_dir))})")

