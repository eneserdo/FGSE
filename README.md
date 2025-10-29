# Real-Time Manipulation Action Recognition with a Factorized Graph Sequence Encoder
<img width="525" height="270" alt="image" src="https://github.com/user-attachments/assets/cb5bb7dc-8ba3-4582-9577-718ad771e04b" />


[[Arxiv version](https://arxiv.org/abs/2503.12034)]

**Our work has been accepted to IEEE/RSJ IROS 2025** 🎉

## Our IROS Poster for Quick Overview
<img width="3179" height="4494" alt="FGSE-Poster-v4" src="https://github.com/user-attachments/assets/910e837f-b371-4a92-8abe-27785fea1e1e" />

## Environment

You can use following codes to create the required environments (may not be inclusive)

```bash
conda create -n pyg_env python=3.8

conda activate pyg_env

conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia

conda install pyg=2.4 -c pyg

pip install enlighten natsort wandb torchmetrics matplotlib seaborn gdown

```

## Training

For data preparation run:

```bash
# Assuming it is the first time you run the code, if not, omit unzip_anyway, fix_null and fix_fps flags
python prepare_dataset.py -tl 25 -ds 3 --unzip_anyway --fix_fps --fix_null
```

Note that this will create data with W=75 and D=3.

For training run:

```bash
python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name F25_L -tl 25 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4 --disable_wandb
```

## Trained Models

The model weights are available at [google drive](https://drive.google.com/drive/folders/1ZJvG4AOasR46GUYNy8NLmVYhsE7wUj3C)

## Example Test Result

An example test result of our action recognition model, FGSE, on an unseen video recorded in our lab. Note that this is an HRC concept video, which means the robot was teleoperated.

https://github.com/user-attachments/assets/02116f53-a9e0-4ed1-b416-4e6fb8444d69
