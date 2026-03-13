# E2Detect: Object Detection from Event Camera via Sparse Feature Pyramid Recovery
This is the official pytorch implementation of the upcoming IEEE ISCAS 2026 paper titled "[E2Detect: Object Detection from Event Camera via Sparse Feature Pyramid Recovery]()".

<br>

<p align="center">
  <img src="figures/overview_e2detect.jpg" alt="Overview E2Detect" width="590"/>
  <br>
  Overall pipeline for the proposed E2Detect system.
</div>

<br>


## Prerequisites
The code was tested on Linux with the following prerequisites:
1. Python 3.13
2. PyTorch 2.7.1 (CUDA 11.8)
3. MATLAB R2021a
4. VLFeat 0.9.21

Remaining libraries are available in [requirements.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/requirements.txt)

## Installation

- Clone this repository
   ```bash
   git clone https://github.com/engrchrishenry/E2Detect.git
   cd E2Detect
   ```

- Create conda environment
   ```bash
   conda create --name e2detect python=3.13
   conda activate e2detect
   ```

- Install dependencies
  1. Install [PyTorch](https://pytorch.org/get-started/locally/).
  2. Install [FFmpeg](www.ffmpeg.org/download.html).
  3. The remaining packages can be installed via:
     ```bash
     pip install -r requirements.txt
     ```
  4. For running MATLAB scripts, you are required to install [VLFeat](https://www.vlfeat.org/download.html).

## Dataset Preparation from Scratch

### Download [REDS 120fps](https://seungjunnah.github.io/Datasets/reds.html)

- Download [REDS 120fps](https://seungjunnah.github.io/Datasets/reds.html) via the download links provided in [reds_120fps_links.txt]()
    ```bash
    wget -i reds_120fps_links.txt -P REDS_120fps/
    ```

- Unzip files

  ```bash
  unzip REDS_120fps/*.zip -d REDS_120fps/
  ```

- [Optional] Delete the .zip files to save storage space.
  ```bash
  rm REDS_120fps/*.zip
  ```

### Preprocessing [REDS 120fps](https://seungjunnah.github.io/Datasets/reds.html) for [ESIM](https://github.com/uzh-rpg/rpg_vid2e/tree/master)

- Preprocess the [REDS 120fps](https://seungjunnah.github.io/Datasets/reds.html) dataset for generating synthetic events via [ESIM](https://github.com/uzh-rpg/rpg_vid2e/tree/master).
   ```bash
   # Preprocess REDS 120fps dataset (training set) for ESIM
   python preprocess_reds_fast.py --data_dir REDS_120fps/train/train_orig/ \
      --out_dir output/reds_for_esim/train/ \
      --fps 120 \
      --resize \
      --img_size 533:300 \
      --cores -1
   
   # Preprocess REDS 120fps dataset (validation set) for ESIM
   python preprocess_reds_fast.py --data_dir REDS_120fps/val/val_orig/ \
      --out_dir output/reds_for_esim/val/ \
      --fps 120 \
      --resize \
      --img_size 533:300 \
      --cores -1
   ```

### Synthetic Event Generation via [ESIM](https://github.com/uzh-rpg/rpg_vid2e/tree/master)
  
- Follow the instructions [here](https://github.com/uzh-rpg/rpg_vid2e/tree/master) to setup ESIM and build the python binding with GPU support. Use a different conda environment with the exact versions of the dependencies reqired to run ESIM with GPU support. Once ESIM is setup:

  ```bash
  # Generate synthetic events for training
  python esim_torch/scripts/generate_events.py --input_dir=<reds_for_esim_train_path> \
    --output_dir=<events_output_path> \
    --contrast_threshold_neg=0.2 \
    --contrast_threshold_pos=0.2 \
    --refractory_period_ns=0
   
   # Generate synthetic events for valdation
   python esim_torch/scripts/generate_events.py --input_dir=<reds_for_esim_val_path> \
    --output_dir=<events_output_path> \
    --contrast_threshold_neg=0.2 \
    --contrast_threshold_pos=0.2 \
    --refractory_period_ns=0
  ```

### Event Voxel and Patch Generation

  - Generate event voxels
    ```bash
    # Voxel generation for training data
    python prep_data_esim_fast.py --events_dir <synthetic_train_events_path> \
      --upsamp_frames_dir <reds_for_esim_train_path> \
      --out_dir <output_path>

    # Voxel generation for validation data
    python prep_data_esim_fast.py --events_dir <synthetic_validation_events_path> \
      --upsamp_frames_dir <reds_for_esim_validation_path> \
      --out_dir <output_path>
    ```

    Usage:

    ```bash
    options:
      -h, --help            show this help message and exit
      --events_dir EVENTS_DIR
                              Path to directory containing ESIM-generated synthetic events
      --upsamp_frames_dir UPSAMP_FRAMES_DIR
                              Path to directory containing upsampled frames
      --out_dir OUT_DIR     Path to output directory
      --bins BINS           Number of bins for event voxel generation
      --dur_sec DUR_SEC     Event window duration in seconds
      --res RES             Event camera resolution (e.g., '533:300')
      --events_th_low EVENTS_TH_LOW
                              Lower threshold for limiting the number of events within an event window. None to ignore.
      --events_th_high EVENTS_TH_HIGH
                              Upper threshold for limiting the number of events within an event window. None to ignore.
      --kp_th KP_TH         Keypoint threshold for rejecting blank frames. None to ignore.
      --sd_th SD_TH         Standard deviation threshold. None to ignore.
      --range_th RANGE_TH   Range (max value - min value) threshold. None to ignore.
      --th_hist TH_HIST     Clipping threshold for histogram plotting. Value between 0 and 100, e.g., 99.9 means clipping at 99.9 percentile.
      --plot                Save plots.
      --cores CORES         Number of cores to use. -1 -> use all cores.
    ```
  
  - Generate patches for training/validation
    ```bash
    # Generate patches for training
    python gen_patches_fast.py --base_path <train_voxels_path> \
      --out_path <output_train_patches_path> \
      --ip_size 533:300 \
      --patch_size  300:300 \
      --cores -1

    # Generate patches for validation
    python prep_data_esim_fast.py --base_path <validation_voxels_path> \
      --out_path <output_validation_patches_path> \
      --ip_size  533:300 \
      --patch_size  300:300 \
      --cores -1
    ```
  
  

 - Generate final patches
  
   ```bash
   # Generate patches for training
   python gen_patches_fast.py --base_path <train_voxels_path> \
      --out_path <output_train_patches_path> \
      --ip_size 533:300 \
      --patch_size 300:300 \
      --cores -1
      
   # Generate patches for validation
   python gen_patches_fast.py --base_path <validation_voxels_path> \
      --out_path <output_validation_patches_path> \
      --ip_size 533:300 \
      --patch_size 300:300 \
      --cores -1
   ```



















### [REDS 120FPS](https://seungjunnah.github.io/Datasets/reds.html)

- Download REDS 120 FPS via the download links provided in [reds_120fps_links.txt]()
    ```bash
    wget -i reds_120fps_links.txt -P REDS_120fps/
    ```

- Unzip files

  ```bash
  unzip REDS_120fps/*.zip -d REDS_120fps/
  ```

- [Optional] Delete the .zip files to save storage space.
  ```bash
  rm REDS_120fps/*.zip
  ```

- Prepare the dataset for generating synthetic events via [ESIM](https://github.com/uzh-rpg/rpg_vid2e/tree/master).
   ```bash
   # Prepare training data for inputting to ESIM
   python prep_reds_for_esim_fast.py --data_dir REDS_120fps/train/train_orig/ \
      --out_dir output/reds_for_esim/train/ \
      --fps 120 \
      --resize \
      --img_size 533:300 \
      --cores -1
   
   # Prepare valdation data for inputting to ESIM
   python prep_reds_for_esim_fast.py --data_dir REDS_120fps/val/val_orig/ \
      --out_dir output/reds_for_esim/val/ \
      --fps 120 \
      --resize \
      --img_size 533:300 \
      --cores -1
   ```

- Synthesize events
  
  Follow the instructions [here](https://github.com/uzh-rpg/rpg_vid2e/tree/master) to setup ESIM and build the python binding with GPU support. Use a different conda environment with the exact versions of the dependencies reqired to run ESIM with GPU support. Once ESIM is setup:

  ```bash
   # Generate training synthetic events
  python esim_torch/scripts/generate_events.py --input_dir=<reds_for_esim_train_path> \
    --output_dir=<events_output_path> \
    --contrast_threshold_neg=0.2 \
    --contrast_threshold_pos=0.2 \
    --refractory_period_ns=0
   
   # Generate valdation synthetic events
   python esim_torch/scripts/generate_events.py --input_dir=<reds_for_esim_val_path> \
    --output_dir=<events_output_path> \
    --contrast_threshold_neg=0.2 \
    --contrast_threshold_pos=0.2 \
    --refractory_period_ns=0
  ```

- Event voxel generation

  ```bash
  # Voxel generation for training data
  python prep_data_esim_fast.py --events_dir <synthetic_train_events_path> \
    --upsamp_frames_dir <reds_for_esim_train_path> \
    --out_dir <output_path>

   # Voxel generation for validation data
   python prep_data_esim_fast.py --events_dir <synthetic_validation_events_path> \
    --upsamp_frames_dir <reds_for_esim_validation_path> \
    --out_dir <output_path>
  ```
  
  Usage:

  ```bash
  options:
   -h, --help            show this help message and exit
   --events_dir EVENTS_DIR
                           Path to directory containing ESIM-generated synthetic events
   --upsamp_frames_dir UPSAMP_FRAMES_DIR
                           Path to directory containing upsampled frames
   --out_dir OUT_DIR     Path to output directory
   --bins BINS           Number of bins for event voxel generation
   --dur_sec DUR_SEC     Event window duration in seconds
   --res RES             Event camera resolution (e.g., '533:300')
   --events_th_low EVENTS_TH_LOW
                           Lower threshold for limiting the number of events within an event window. None to ignore.
   --events_th_high EVENTS_TH_HIGH
                           Upper threshold for limiting the number of events within an event window. None to ignore.
   --kp_th KP_TH         Keypoint threshold for rejecting blank frames. None to ignore.
   --sd_th SD_TH         Standard deviation threshold. None to ignore.
   --range_th RANGE_TH   Range (max value - min value) threshold. None to ignore.
   --th_hist TH_HIST     Clipping threshold for histogram plotting. Value between 0 and 100, e.g., 99.9 means clipping at 99.9 percentile.
   --plot                Save plots.
   --cores CORES         Number of cores to use. -1 -> use all cores.
  ```

 - Generate final patches
  
   ```bash
   # Generate patches for training
   python gen_patches_fast.py --base_path <train_voxels_path> \
      --out_path <output_train_patches_path> \
      --ip_size 533:300 \
      --patch_size 300:300 \
      --cores -1
      
   # Generate patches for validation
   python gen_patches_fast.py --base_path <validation_voxels_path> \
      --out_path <output_validation_patches_path> \
      --ip_size 533:300 \
      --patch_size 300:300 \
      --cores -1
   ```