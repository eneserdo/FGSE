# Real-Time Manipulation Action Recognition with a Factorized Graph Sequence Encoder

(The CoAx branch)

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
# Assuming it is the first time you run the code, if not, omit unzip_anyway
python prepare_dataset.py -tl 40 -ds 1 --unzip_anyway
```

Note that this will create data with W=40 and D=1.

For training run:

```bash
python cross_validation.py -lr 0.001 -wd 0.0005 -me 60 --scheduler_step_size -1 --saving_freq 2 --name NCF40d1_pl_dr30 -tl 40 --downsample 1 --merged_pred late --temporal_type tr --norm_layer LayerNorm --edge_dropout 0.30 --weighted_mv auto --disable_wandb
```


## Trained Models

Once uploaded, the models will be available at [google drive](https://drive.google.com/drive/folders/1ZJvG4AOasR46GUYNy8NLmVYhsE7wUj3C)