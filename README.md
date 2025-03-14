# Real-Time Manipulation Action Recognition with a Factorized Graph Sequence Encoder

Work in progress...

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
python cross_validation.py -lr 0.001 -wd 0.0005 -me 40 --scheduler_step_size 8 --saving_freq 2 --name F25_L -tl 25 --downsample 3 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.15 --weighted_mv auto --dist_threshold 0.4
```


## Trained Models

Once uploaded, the models will be available at [google drive](https://drive.google.com/drive/folders/1ZJvG4AOasR46GUYNy8NLmVYhsE7wUj3C)