# E2Detect
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

### [REDS 120FPS](https://seungjunnah.github.io/Datasets/reds.html)

- Download REDS 120 FPS via the download links provided in [reds_120fps_links.txt]()
    ```bash
    wget -i reds_120fps_links.txt -P REDS_120fps
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
      --img_size 320:180 \
      --cores -1
   
   # Prepare valdation data for inputting to ESIM
   python prep_reds_for_esim_fast.py --data_dir REDS_120fps/val/val_orig/ \
      --out_dir output/reds_for_esim/val/ \
      --fps 120 \
      --resize \
      --img_size 320:180 \
      --cores -1
   ```

- Synthesize events
  
  Follow the instructions [here](https://github.com/uzh-rpg/rpg_vid2e/tree/master) to setup ESIM and build the python binding with GPU support. Use a different conda environment with the exact versions of the dependencies reqired to run ESIM with GPU support. Once ESIM is setup:

  Generate synthetic events via [generate_events.py](https://github.com/uzh-rpg/rpg_vid2e/blob/master/esim_torch/scripts/generate_events.py). Sample command:

  ```bash
   # Generate training synthetic events
  python esim_torch/scripts/generate_events.py --input_dir=output/reds_for_esim/train/ \
    --output_dir=output/reds_events/train/ \
    --contrast_threshold_neg=0.2 \
    --contrast_threshold_pos=0.2 \
    --refractory_period_ns=0
   
   # Generate valdation synthetic events
   python esim_torch/scripts/generate_events.py --input_dir=output/reds_for_esim/val/ \
    --output_dir=output/reds_events/val/ \
    --contrast_threshold_neg=0.2 \
    --contrast_threshold_pos=0.2 \
    --refractory_period_ns=0
  ```

  