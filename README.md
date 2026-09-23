# SACL4ASD
Subsample-aware acoustic representation learning with
angular margins for robust anomalous sound detection

## Citation
If you use this code in your research, please cite our paper:

## Overview
This repository provides a complete end-to-end pipeline for robust anomalous sound detection:
- A subsample-aware background exchange strategy improves sample diversity.
- ArcFace-enhanced contrastive learning improves feature discrimination.
-  A paired subsample fusion strategy is introduced for prototype-based anomaly scoring.

## Project Structure

├── .gitattributes        # Git configuration for file attributes and repository management

├── config.py             # Configuration utilities for loading and managing experiment parameters

├── config.yaml           # Global experiment configuration, including model and training settings

├── dataset           # subsample-aware and Back-Ex

├── tools           # train, fusion loss, and pair subsample fusion

├── model             # Implementation of the proposed model and its main components

├── da             # data augmentation

└── README.md             # Project overview, environment setup, usage, and reproduction instructions

More files will be added in the future.

## Requirements
We use Conda python 3.8+ and strongly recommend that you create a new environment.
* Prerequisite: Python 3.8 or higher versions
```shell script
conda create -n MyEnv python=3.8
conda activate MyEnv
```

## Environment
This code is tested using Python 3.8, Pytorch 1.10, and CUDA 11.1
* Install all packages in the requirement.txt
```shell script
pip3 install -r requirements.txt
```

## Datasets
### DCASE 2022 
More details can be find in this [link](https://dcase.community/challenge2022/index). please request and download the data from the original WORKSHOP.

### DCASE 2024 
More details can be find in this [link](https://dcase.community/challenge2024/index). please request and download the data from the original WORKSHOP.


## Quick Start
1. Configure Datasets
Place your audio datasets in .wav format in the ./data

2. Update config.py to add your dataset paths:

3. Run the Full Pipeline:
python train.py

4. Output Results
All results are automatically saved in the ./results/{dataset_name}/ directory:


## Get Involved
Should you have any query please contact me.
Please create a GitHub issue if you have any questions, suggestions, requests or bug-reports. 
Don't hesitate to send us an e-mail or report an issue, if something is broken or if you have further questions.
