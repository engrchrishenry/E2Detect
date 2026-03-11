# E2Detect
This is the official pytorch implementation of the upcoming IEEE ISCAS 2026 paper titled "[E2Detect: Object Detection from Event Camera via Sparse Feature Pyramid Recovery]()".

<br>

<p align="center">
  <img src="figures/overview_e2detect.jpg" alt="Overview E2Detect" width="590"/>
  <br>
  Overall workflow of the proposed E2SIFT pipeline.
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

