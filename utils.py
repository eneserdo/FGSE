import argparse
import glob
import json
import os
import random
import shutil
import warnings
from math import ceil
from os.path import join as opj
from typing import List

import matplotlib.pyplot as plt
import natsort
import numpy as np
import seaborn as sns
import torch
from torchmetrics.classification import (MulticlassConfusionMatrix,
                                         MulticlassF1Score)

# functorch was merged into pytorch after certain version. I used try-except to make it compatible with both versions
try: 
    torch_vmap = torch.vmap
except AttributeError:
    import functorch
    torch_vmap = functorch.vmap

from collections import Counter

from scipy.optimize import linear_sum_assignment

from dataset import ACTIONS
from metrics import f1_at_k_single_example
from visualization_utils import (plot_segmentation, plot_segmentation_partial,
                                 plot_segmentation_v2)

plt.ioff()


def arg_parser_base():
    ap = argparse.ArgumentParser()

    # Dataset
    ap.add_argument("-rt", "--root", type=str, default='./coax', help="Root directory of dataset.")
    ap.add_argument("-p", "--process_it", action="store_true", help="...")
    ap.add_argument("-bs", "--batch_size", type=int, default=256, help="...")
    ap.add_argument("-tl", "--temporal_length", type=int, default=10, help="...")
    ap.add_argument("-ts", "--test_subject", type=int, default=1, choices=[1, 2, 3, 4, 5, 6], help="...")
    ap.add_argument("-vt", "--validation_take", type=int, default=7, choices=[i for i in range(10)], help="...")
    ap.add_argument("--downsample", type=int, default=1, help="Downsampling ratio")
    ap.add_argument("--use_vf", action="store_true", help="use visual features")
    ap.add_argument("--monograph", action="store_true", help="use monograph")
    ap.add_argument("--filtered_data", action="store_true", help="use filtered data")

    # Model - GNN part
    ap.add_argument("-g", "--gnn_type", type=str, default="TransformerConv",
                    choices=["NNConv", "GATConv", "TransformerConv", "GATv2Conv"], help="...")

    ap.add_argument("-ch", "--channels", type=int, nargs="+", default=[64, 64], help="...")
    ap.add_argument("-nh", "--num_heads", type=int, nargs="+", default=[2, 2], help="...")
    ap.add_argument("-nl", "--num_layers", type=int, default=2, help="...")
    ap.add_argument("--no_concat", action="store_true", help="Uses average instead of concat in gnn")

    ap.add_argument("--saving_freq", type=int, default=4, help="...")
    ap.add_argument("--use_global_pooling", action="store_true", help="...")    # :(
    ap.add_argument("--edge_dropout", type=float, default=0.0, help="...")
    ap.add_argument("--dropout", type=float, default=0.0, help="...")
    ap.add_argument("--norm_layer", type=str, default="LayerNorm", choices=["BatchNorm", "InstanceNorm", "LayerNorm", "GraphNorm", "PairNorm", "none"], help="Normalization  for GNN layer")
    ap.add_argument("--use_pos", action="store_true", help="Use positions too as node features")
    ap.add_argument("--use_embedding", action="store_true", help="Instead of one-hot encoded node features, use embedding layer")
    ap.add_argument("--use_pos_edge", action="store_true", help="Use positions too as edge features")
    ap.add_argument("--use_embedding_edge", action="store_true", help="Instead of one-hot encoded edge features, use embedding layer")
    ap.add_argument("--dist_threshold", type=float, default=None, help="...")
    

    # Model - Temporal part
    ap.add_argument("--dropout_tr", type=float, default=0.1, help="...")
    ap.add_argument("-hs", "--lstm_hidden_size", type=int, default=128, help="...")
    ap.add_argument("--temporal_type", type=str, default="tr", choices=["lstm", "bi", "tr", "edtr", "none"], help="...")
    ap.add_argument("--norm_first", action="store_true", help="To make norm_first=true in TransformerEncoderLayer")
    ap.add_argument("--pos_enc", type=str, default="original", choices=["original", "learnable", "none"], help="Type of positional encoding")
    ap.add_argument("--num_heads_tr", type=int, default=4, help="...")
    ap.add_argument("--num_of_temp_layers", type=int, default=2, help="...")
    ap.add_argument("--causal", action="store_true", help="Use causal attention in transformer (Available when whole video processing)")
    
    # Model - Others
    ap.add_argument("--dim_feedforward", type=int, default=0, help="transformer encoder layer's dim_feedforward, if 0, it is set to lstm_hidden_size//4*3")
    ap.add_argument("--merged_pred", type=str, default="none", choices=["early", "late", "none", "attention"], help="Merge hands predictions")


    # Optimizer
    ap.add_argument("-o", "--optimizer_type", type=str,
                    choices=["Adam", "AdamW", "NAdam", "RMSprop", "SGD"],
                    help="Type of optimizer to be used  from torch.optim", default='Adam')
    ap.add_argument("-lr", "--learning_rate", type=float, default=0.0005, help="...")
    ap.add_argument("-mn", "--momentum", type=float, default=0.9, help="...")
    ap.add_argument("-wd", "--weight_decay", type=float, default=0.0005, help="...")
    ap.add_argument("--scheduler_step_size", type=int, default=-1, help="...")
    ap.add_argument("--scheduler_gamma", type=float, default=0.5, help="...")
    ap.add_argument("--weighted_loss", type=str, default="none", choices=["none", "inv", "sqrt_inv", "effective"],
                    help="Use class frequencies as weight for loss function")       # :(
    ap.add_argument("--smoothing", type=float, default=0.0, help="Label smoothing weight")

    ap.add_argument("--num_acc", type=int, default=1, help="Number of grad accumulation steps")
    ap.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    ap.add_argument("--num_workers", type=int, default=5, help="Number of workers for dataloader")


    # Metrics
    ap.add_argument("--weighted_mv", type=str, default="none", choices=["none", "w1", "w2", "auto"], help="...")

    # Training
    ap.add_argument("-me", "--max_epochs", type=int, default=20, help="Max number of epochs")
    ap.add_argument("-r", "--restore", type=int, help="...")
    ap.add_argument("-rn", "--restored_name", type=str, help="...")
    ap.add_argument("-fih", "--filter_idle_hold", action="store_true", help="...")

    # Logging
    ap.add_argument("-n", "--name", required=True, type=str, help="Name of experiment for logging into Wandb")
    ap.add_argument("--disable_wandb",action="store_true", help="Disable wandb")
    ap.add_argument("--save_incorrects",action="store_true", help="saves some incorrect samples")
    ap.add_argument("--save_individual",action="store_true", help="Visualize individual segment predictions")
    ap.add_argument("--save_segments",action="store_true", help="Visualize final predictions for whole video")

    # Others
    ap.add_argument("--gpu_id", type=int, default=0, help="...")
    ap.add_argument("--seed", type=int, default=34, help="...")
    ap.add_argument("--return_attention", action="store_true", help="...")

    return ap

def arg_parser():
    ap = arg_parser_base()
    return ap.parse_args()

def arg_parser_cv():
    ap = arg_parser_base()

    # Cross validation 
    ap.add_argument("--num_folds", type=int, default=6, help="Number of folds")
    ap.add_argument("--num_sweeps", type=int, default=0, help = "Number of sweep")
    
    return ap.parse_args()


# def arg_parser_viz_attn():
#     ap = argparse.ArgumentParser()

#     # Dataset
#     ap.add_argument("-rt", "--root", type=str, default='.', help="Root directory of dataset.")
#     ap.add_argument("-p", "--process_it", action="store_true", help="...")
#     ap.add_argument("-tl", "--temporal_length", type=int, default=10, help="...")
#     ap.add_argument("-ts", "--test_subject", type=int, default=1, choices=[1, 2, 3, 4, 5, 6], help="...")
#     ap.add_argument("-vt", "--validation_take", type=int, default=7, choices=[i for i in range(10)], help="...")
#     ap.add_argument("--downsample", type=int, default=1, help="Downsampling ratio")
#     ap.add_argument("--use_vf", action="store_true", help="use visual features")
#     ap.add_argument("--monograph", action="store_true", help="use monograph")

#     # Model - GNN part
#     ap.add_argument("-g", "--gnn_type", type=str, default="TransformerConv",
#                     choices=["NNConv", "GATConv", "TransformerConv", "GATv2Conv"], help="...")
#     ap.add_argument("-hc", "--hidden_channels", type=int, default=64, help="...")
#     ap.add_argument("-oc", "--out_channels", type=int, default=64, help="...")
#     ap.add_argument("-nh", "--num_heads", type=int, default=2, help="...")
#     ap.add_argument("-nl", "--num_layers", type=int, default=2, help="...")
#     ap.add_argument("--use_global_pooling", action="store_true", help="...")    # :(
#     ap.add_argument("--edge_dropout", type=float, default=0.0, help="...")
#     ap.add_argument("--dropout", type=float, default=0.0, help="...")
#     ap.add_argument("--norm_layer", type=str, default="LayerNorm", choices=["BatchNorm", "InstanceNorm", "LayerNorm", "GraphNorm", "PairNorm", "none"], help="Normalization  for GNN layer")
#     ap.add_argument("--use_pos", action="store_true", help="Use positions too as node features")
#     ap.add_argument("--use_embedding", action="store_true", help="Instead of one-hot encoded node features, use embedding layer")
    

#     # Model - Temporal part
#     ap.add_argument("--dropout_tr", type=float, default=0.1, help="...")
#     ap.add_argument("-hs", "--lstm_hidden_size", type=int, default=128, help="...")
#     ap.add_argument("--temporal_type", type=str, default="tr", choices=["lstm", "bi", "tr", "edtr", "none"], help="...")
#     ap.add_argument("--norm_first", action="store_true", help="To make norm_first=true in TransformerEncoderLayer")
#     ap.add_argument("--pos_enc", type=str, default="original", choices=["original", "learnable", "none"], help="Type of positional encoding")
#     ap.add_argument("--num_heads_tr", type=int, default=4, help="...")
#     ap.add_argument("--num_of_temp_layers", type=int, default=2, help="...")
    
#     # Model - Others
#     ap.add_argument("--dim_feedforward", type=int, default=0, help="transformer encoder layer's dim_feedforward, if 0, it is set to lstm_hidden_size//4*3")
#     ap.add_argument("--merged_pred", type=str, default="none", choices=["early", "late", "none", "attention"], help="Merge hands predictions")


#     # Others
#     ap.add_argument("--gpu_id", type=int, default=0, help="...")
#     ap.add_argument("--seed", type=int, default=34, help="...")
    
#     # save_dir
#     ap.add_argument("--save_dir", default="debug", type=str, help="...")  # , default="raw"
    
#     # model_path
#     ap.add_argument("--model_path", type=str, default=None, help="...")

#     # discard_ratio
#     ap.add_argument("--discard_ratio", type=float, default=0.1, help="...")

#     # fuse method
#     ap.add_argument("--fuse_method", type=str, default="mean", choices=["mean", "min", "max"], help="...")    

#     args = ap.parse_args()
#     args.return_attention = True
#     args.filter_idle_hold = False

#     return args 

# def arg_parser_inference():
#     ap = argparse.ArgumentParser()
        
#     ap.add_argument("--raw_dir", default="raw", type=str, help="...") 
#     ap.add_argument("--save_dir", default="tests", type=str, help="...")  # , default="raw"

#     ap.add_argument("-s", "--test_subject", type=int, required=True, help="...") 
#     # ap.add_argument("-gt", "--ground_truth_path", help="...")
    
#     ap.add_argument("-tl", "--temporal_length", type=int, default=10, help="...")
#     ap.add_argument("-bs", "--batch_size", type=int, default=256, help="...")

#     # Model
#     ap.add_argument("-g", "--gnn_type", type=str, required=True, choices=["NNConv", "GATConv", "TransformerConv", "GATv2Conv"], help="...")
#     ap.add_argument("--temporal_type", type=str, default="lstm", choices=["lstm", "bi", "tr"], help="...")
#     ap.add_argument("-hc", "--hidden_channels", type=int, default=64, help="...")
#     ap.add_argument("-oc", "--out_channels", type=int, default=64, help="...")
#     ap.add_argument("-hs", "--lstm_hidden_size", type=int, default=128, help="...")
#     ap.add_argument("-nh", "--num_heads", type=int, default=2, help="...")
#     ap.add_argument("--saving_freq", type=int, default=5, help="...")
#     ap.add_argument("-nbn", "--no_bn", action="store_true", help="...")
#     ap.add_argument("--use_global_pooling", action="store_true", help="...")
#     ap.add_argument("--edge_dropout", type=float, default=0.0, help="...")
#     ap.add_argument("--dropout", type=float, default=0.0, help="...")
#     ap.add_argument("--use_cls_token", action="store_true", help="Use class token in transformer")


#     # Restore
#     ap.add_argument("-r", "--restore", type=int, required=True, help="...")
#     ap.add_argument("-rn", "--restored_name", type=str, required=True, help="...")
#     ap.add_argument("-fih", "--filter_idle_hold", action="store_true", help="...")

#     return ap.parse_args()




def get_minutes(elapsed) -> str:
    return f"{round((elapsed) // 60)}.{round((elapsed) % 60)}"


def plot_conf_mat(mat, name, do_save=False, do_return_plot=False):
    if isinstance(mat, torch.Tensor):
        mat = mat.cpu().numpy()
    
    mat_long = np.round(mat * 100).astype(int)

    plt.figure(figsize=(14, 12))
    sns.heatmap(mat_long, square=True, annot=True, cmap='Blues', cbar=False, xticklabels=ACTIONS, yticklabels=ACTIONS,
                vmin=0, vmax=100)
    plt.yticks(rotation=0)
    plt.title(f"{name}")
    plt.xlabel('Predicted')
    plt.ylabel('Ground truth')
    if do_save:
        raise NotImplementedError("conf_mat diye bir klasor yok artik")
        plt.savefig(opj("conf_mat", f'{name}.png'))
    if do_return_plot:
        return plt
    else:
        plt.close()

def plot_line(line, name, do_save=False, do_return_plot=False):
    
    # assert len(line.shape)==1, "Line must be 1D"

    line = np.array(line)*100

    plt.figure(figsize=(14, 12))
    plt.plot(line)
    plt.yticks(np.arange(min(line)-1, max(line)+1, 1.0))
    plt.title(name)
    plt.axhline(y = np.max(line), linestyle = '--') 
    plt.xlabel('t')
    plt.ylabel('F1')
    if do_save:
        raise NotImplementedError("conf_mat diye bir klasor yok artik")
        plt.savefig(opj("conf_mat", f'{name}.png'))
    if do_return_plot:
        return plt
    else:
        plt.close()


def create_ground_truth_segments(label_series):
    change_occurring = torch.concat([label_series[1:] - label_series[:-1], torch.zeros(size=(1, label_series.shape[-1]), device=label_series.device)], dim=0)
    return torch.where(change_occurring != 0, 1.0, 0.0) # Should this be of shape [N, B, 2] instead of [N, B*2]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def flatten_seq(x):
        """
            input:
                x -> tensor of shape (n, b*2, f)
            output:
                y -> flattened tensor of shape (n*b*2, f)
        """
        return torch.flatten(torch.transpose(x, 0, 1), start_dim=0, end_dim=1)

def get_filters_for_imbalance(dataBatchList):
    """ 2 out of 3 samples for the actions idle and hold were discarded """

    batch_size = dataBatchList[0].y.shape[0]
    left_filter = torch.full((len(dataBatchList), batch_size // 2), True)  # shape = [sequence_length, batch_size]
    right_filter = torch.full((len(dataBatchList), batch_size // 2), True)  # shape = [sequence_length, batch_size]

    for i in range(batch_size // 2):
        count_idle_left = 0
        count_idle_right = 0
        count_hold_left = 0
        count_hold_right = 0

        for t, dataBatch in enumerate(dataBatchList):
            count_idle_left = (count_idle_left + torch.argmax(dataBatch[i].y[0]) == ACTIONS.index("idle")) % 3
            count_idle_right = (count_idle_right + torch.argmax(dataBatch[i].y[1]) == ACTIONS.index("idle")) % 3
            count_hold_left = (count_hold_left + torch.argmax(dataBatch[i].y[0]) == ACTIONS.index("hold")) % 3
            count_hold_right = (count_hold_right + torch.argmax(dataBatch[i].y[1]) == ACTIONS.index("hold")) % 3

            left_filter[t, i] = not (count_idle_left in [1, 2])
            right_filter[t, i] = not (count_idle_right in [1, 2])

            left_filter[t, i] = not (count_hold_left in [1, 2])
            right_filter[t, i] = not (count_hold_right in [1, 2])

    return left_filter, right_filter


def align_embeddings_to_right(x, mask):
    """
        Ref 1: https://discuss.pytorch.org/t/reorder-non-zero-values-in-a-2d-matrix/112314
        Ref 2: https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
    """

    emd_size = x.shape[2]
    mask = mask[:, :, None]

    # Somehow when it is on gpu it gives wrong values, to fix it I used "stable=True"
    # Still suspicious
    index = torch.sort(~mask.repeat(1, 1, emd_size).to(x.device) * 1, dim=0, descending=True, stable=True)[1]
    y = x.gather(0, index)

    # d_mask = torch.sum(mask, dim=0, keepdims=True) > torch.arange(x.shape[0]).reshape(-1,1)
    # y = torch.zeros_like(x)
    # y[d_mask] = x[mask]

    return y


def debug_fih(dataBatchList, b):
    for i in range(10):

        # hold
        if torch.argmax(dataBatchList[i].y[b]).item() == 5:
            print("h ", end="")
        # idle
        elif torch.argmax(dataBatchList[i].y[b]).item() == 0:
            print("i ", end="")
        else:
            print(torch.argmax(dataBatchList[i].y[b]).item(), end=" ")

    print("")


def print_non_uniform_ys(dataBatchList):
    bs = dataBatchList[0].y.shape[0]
    for b in range(bs):
        act = torch.argmax(dataBatchList[0].y[b]).item()
        for i in range(10):
            if act != torch.argmax(dataBatchList[i].y[b]).item():
                print("b id: ", b)
                debug_fih(dataBatchList, b)
                break


def debug_align(device="cpu"):
    mask = torch.full((4, 3), True)
    mask[0, 0] = False
    mask[2, 2] = False
    mask[3, 0] = False
    mask[3, 1] = False

    # print("mask:\n", mask)

    x = torch.arange(12).reshape(4, 3, 1)
    x = x.repeat(1, 1, 2)
    x[~mask] = 0

    x = x.to(device)
    mask = mask.to(device)

    # t=4, bs=3, emb_size=2

    # print("x:\n", x)
    # print("x[:, :, 0]:\n", x[:, :, 0])

    # y = align_embeddings_to_right(x, mask)
    emd_size = x.shape[2]
    mask = mask[:, :, None]

    # Somehow when it is on gpu it gives wrong values
    index = torch.sort(~mask.repeat(1, 1, emd_size).to(x.device) * 1, dim=0, descending=True)[1]
    y = x.gather(0, index)

    # print("y[:, :, 0]:\n", y[:, :, 0])

    return x, y, index, mask


def get_relations(base_dir, subject, task, take, i):
    """ Returns a set for relations in the frame """

    # derived_dir=base_dir+"bimacs_derived_data/"
    derived_dir = opj(base_dir, "bimacs_derived_data")

    _3d_objects_dirs = opj(derived_dir, subject, task, take, "3d_objects")
    _3d_objects_json_list = natsort.natsorted(glob.glob(opj(_3d_objects_dirs, "*.json")))

    relation_json_dirs = opj(derived_dir, subject, task, take, "spatial_relations")
    relations_path = natsort.natsorted(glob.glob(opj(relation_json_dirs, "*.json")))

    with open(relations_path[i]) as f:
        relations = json.load(f)

    with open(_3d_objects_json_list[i]) as f:
        _3d = json.load(f)

    # log=""
    # text=""
    readable_relations = []

    for relation in relations:
        oi = relation["object_index"]
        si = relation["subject_index"]

        # log = log + _3d[oi]["class_name"]+ " - " + relation["relation_name"] + " - " + _3d[si]["class_name"] +"; "
        # log = log + _3d[oi]["class_name"]+ " - " + relation["relation_name"] + " - " + _3d[si]["class_name"] +"; "

        readable_relations.append(
            _3d[oi]["class_name"] + " - " + relation["relation_name"] + " - " + _3d[si]["class_name"])

        # if relation["relation_name"]=="contact":

        # text=text + f"({oi})"+object_mapping[oi] + " - " + f"({si})"+ object_mapping[si] + "\n"
        # if oi < si:
        #     text=text + _3d[oi]["class_name"]+ " - " + _3d[si]["class_name"] +"\n"

    # print("FRAME:\n", log, "\n")
    return readable_relations


def visualize_incorrect_preds(preds, labels, idxes, test_idx_sample_mapping, graph_dataset_dir="raw",
                              rgb_dataset_dir="/home/stannis/enes/bimanual/bimacs_rgbd_data"):
    assert os.path.exists(rgb_dataset_dir), "RGB dataset directory does not exist"
    assert os.path.exists(graph_dataset_dir), "Graph dataset directory does not exist"

    raise NotImplementedError("Not implemented")
    save_dir_name = "incorrect_predictions"
    # if os.path.exists(save_dir_name):
    # print("Removing existing", save_dir_name)
    # shutil.rmtree(save_dir_name)
    os.makedirs(save_dir_name, exist_ok=True)

    incorrect_mask = preds != labels
    preds, labels, idxes = preds[incorrect_mask], labels[incorrect_mask], idxes[incorrect_mask]

    num_of_samples = 10
    random_samples = np.random.randint(0, len(preds), num_of_samples)
    preds, labels, idxes = preds[random_samples], labels[random_samples], idxes[random_samples]

    for idx, p, gt in zip(idxes, preds, labels):
        subject, task, take, start_frame, end_frame, is_mirrored = test_idx_sample_mapping[idx]
        sample_save_dir = opj(save_dir_name, f"{ACTIONS[p]}_{ACTIONS[gt]}_{idx}")
        os.mkdir(sample_save_dir)

        # Copy frames
        for frame in range(start_frame, end_frame + 1):
            chunk_no = frame // 100
            frame_path = opj(rgb_dataset_dir, subject, task, take, "rgb", f"chunk_{chunk_no}", f"frame_{frame}.png")
            shutil.copy(frame_path, sample_save_dir)

        # Open a text file, and save relations
        with open(opj(sample_save_dir, "relations.txt"), "w") as f:
            prev_relations = []
            for frame in range(start_frame, end_frame + 1):
                f.write(f"Frame {frame}:\n\n")

                curr_relations = get_relations(graph_dataset_dir, subject, task, take, frame)

                diff_relations = list(set(curr_relations) - set(prev_relations))

                for relation in diff_relations:
                    f.write(relation + "; ")
                f.write("\n")

                prev_relations = curr_relations

def calculate_boundaries(arr):
    """ Sparse array to dense boundaries """

    id_changes = torch.where(arr ==1)[0]
    boundaries = torch.zeros((len(id_changes), 2), dtype=torch.int32)

    boundaries[1:, 0] = id_changes[:-1]+1
    boundaries[:, 1] = id_changes

    return boundaries


def calc_cost_matrix_single(gt_seg, pred_seg):
    """ Calculates cost matrix using hungarian algorithm for single example """
    # old:
    # gt_seg_pseudo_label=np.cumsum(gt_seg)
    # pred_seg_pseudo_label=np.cumsum(pred_seg)

    # stacked=np.stack((gt_seg_pseudo_label, pred_seg_pseudo_label), axis=1)
    # stacked = stacked.tolist()
    # stacked= [tuple(s) for s in stacked]
    # cntr=Counter(stacked)
    # m=sparse.csc_matrix((list(cntr.values()), list(cntr.keys())))
    # return m.toarray()
    # new: 
    # TODO add assert for gt_seg and pred_seg shapes 
    ground_truth_boundaries = calculate_boundaries(gt_seg)
    predicted_boundaries = calculate_boundaries(pred_seg)
    
    cost_matrix = torch.zeros((len(ground_truth_boundaries), len(predicted_boundaries)))

    for i, (gt_start, gt_end) in enumerate(ground_truth_boundaries):
        for j, (p_start, p_end) in enumerate(predicted_boundaries):

            # it is already passed
            # g: [+---+      ]
            # p: [       +--+]
            if gt_end < p_start:
                break

            # it is not reached yet
            # g: [       +--+]
            # p: [+---+      ]
            elif p_end < gt_start:
                continue

            # g: [ +----------+  ]
            # p: [    +---+      ]
            elif gt_start <= p_start and p_end <= gt_end:
                # cost_matrix[i, j] = p_end - p_start + 1
                cost_matrix[i, j] = (p_end - p_start + 1) / (gt_end - gt_start + 1)
            
            # g: [    +---+      ]
            # p: [ +----------+  ]
            elif p_start <= gt_start and gt_end <= p_end:
                # cost_matrix[i, j] = gt_end - gt_start + 1
                cost_matrix[i, j] = (gt_end - gt_start + 1) / (p_end - p_start + 1)
            
            # g: [   +--------+   ]
            # p: [ +----+         ]
            # or
            # g: [ +----+         ]
            # p: [   +--------+   ] 
            else:
                arg1 = p_end - gt_start + 1
                arg2 = gt_end - p_start + 1
                assert arg1 > 0 and arg2 > 0, "arg1 and arg2 must be positive"
                # cost_matrix[i, j] = min(max(arg1, 0), max(arg2, 0)) 
                cost_temp = arg1/arg2 if arg1 < arg2 else arg2/arg1
    
    return cost_matrix, ground_truth_boundaries, predicted_boundaries
    

def assign_seg_labels_single(cost_matrix, gt_seg_labels, gt_boundaries, pred_boundaries):
    """ Assigns ground truth segment labels to predicted unknown segments for single example """

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    pred_seg_labels = torch.zeros_like(gt_seg_labels)

    for i, j in zip(row_ind, col_ind):
        label_loc, _ = gt_boundaries[i]
        label = gt_seg_labels[label_loc]
        start, end = pred_boundaries[j]
        pred_seg_labels[start:end] = label

    return pred_seg_labels


def assign_seg_labels(gt_seg_labels, gt_seg, pred_seg):
    """ Assigns ground truth segment labels to predicted unknown segments """
    # gt_seg_labels: [T, B, C]
    # gt_seg: [T, B] (ones and zeros, ones indicate segment endings)
    # pred_seg: [T, B] (ones and zeros, ones indicate segment endings)
    assert gt_seg_labels.shape[:2] == gt_seg.shape == pred_seg.shape, "Shapes must be equal"

    # calculate cost matrix
    gt_seg_labels = gt_seg_labels.argmax(axis=2)
    pred_seg = pred_seg > 0.5
    
    # FIXME: this is a hack to make sure that the last frame is always a segment ending
    gt_seg[-1, :] = 1
    pred_seg[-1, :] = 1
    
    batch_size=gt_seg.shape[1]
    pred_seg_labels = torch.zeros_like(gt_seg_labels)

    for b_id in range(batch_size):
        gt_seg_labels_b = gt_seg_labels[:, b_id]
        gt_seg_b = gt_seg[:, b_id]
        pred_seg_b = pred_seg[:, b_id]

        # calculate cost matrix
        cost_matrix, gt_boundaries, pred_boundaries = calc_cost_matrix_single(gt_seg_b, pred_seg_b)

        # apply hungarian algorithm
        # assign labels to unknown segments
        pred_seg_labels_b = assign_seg_labels_single(cost_matrix, gt_seg_labels_b, gt_boundaries, pred_boundaries)
        pred_seg_labels[:, b_id] = pred_seg_labels_b

    return pred_seg_labels


class F1_at_k:
    # Very custom class for calculating f1@k
    def __init__(self, num_classes, batch_mode=False):
        self.num_classes = num_classes
        self.k_10 = []
        self.k_25 = []
        self.k_50 = []
        
        # To prevent bugs
        self.batch_mode = batch_mode

        
    def __call__(self, y_true, y_pred, segment_name = None):

        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have same shape"
        
        if self.batch_mode:
            for y_t, y_p in zip(y_true, y_pred):
                self.k_10.append(f1_at_k_single_example(y_t, y_p, num_classes=self.num_classes, overlap=0.1))
                self.k_25.append(f1_at_k_single_example(y_t, y_p, num_classes=self.num_classes, overlap=0.25))
                self.k_50.append(f1_at_k_single_example(y_t, y_p, num_classes=self.num_classes, overlap=0.5))

        else:
            self.k_10.append(f1_at_k_single_example(y_true[:, 0], y_pred[:, 0], num_classes=self.num_classes, overlap=0.1))
            self.k_25.append(f1_at_k_single_example(y_true[:, 0], y_pred[:, 0], num_classes=self.num_classes, overlap=0.25))
            self.k_50.append(f1_at_k_single_example(y_true[:, 0], y_pred[:, 0], num_classes=self.num_classes, overlap=0.5))

    def compute(self):
        return np.mean(self.k_10), np.mean(self.k_25), np.mean(self.k_50)


def majority_voting(arr, total_len, buffer_len, weighted):
    # TODO: add asserts
    # majority voting for overlapping windows for single sample
    # arr: [T,]

    # Example:
    # arr = np.array([[1, 2, 3], 
    #                    [2, 3, 4], 
    #                       [3, 4, 5], 
    #                          [4, 5, 6]])
    # [1...6] <= majority_voting(arr)
    num_arr = arr.shape[0]//buffer_len 
    assert num_arr + buffer_len - 1 == total_len, f"Something is wrong, arr shape: {arr.shape}, total_len: {total_len}, buffer_len: {buffer_len}"  
    # assert num_arr > buffer_len, f"Current implementation only works if num_arr must be greater than buffer_len, num_arr: {num_arr}, buffer_len: {buffer_len}"

    result = np.zeros((total_len), dtype=np.int32)
    confidences = np.zeros((total_len))
    shift = buffer_len - 1

    if weighted != "none":
        # w1 and w2 are only for buffer_len=30
        if weighted == "w1":
            vote_weights = np.array([0]*1 + [1]*1 + [2]*1 + [5]*2 + [6]*3 + [7]*4 + [8]*6 +[7]*4 + [6]*3 + [5]*2 + [2]*1 + [1]*1 + [0]*1)
        elif weighted == "w2":
            vote_weights = np.array([0]*3 + [4]*2 + [5]*3 + [6]*3 + [7]*3 + [8]*2 + [7]*3 +[6]*3 + [5]*3 + [4]*2 + [0]*3)
        elif weighted == "auto":
            # 4 ... 8 ... 4
            half = np.linspace(4, 8, buffer_len//2).round().astype(int)
            if buffer_len % 2 == 0:
                vote_weights = np.concatenate([half, half[::-1]])
            else:
                vote_weights = np.concatenate([half, [8], half[::-1]]) 
        else:
            raise ValueError("weighted must be one of 'none', 'w1', 'w2', or 'auto'")

        # 7//2 -> 3 but -7//2 -> -4
        assert (vote_weights[:buffer_len//2] == vote_weights[-(buffer_len//2):][::-1]).all(), "vote_weights is supposed to be symmetric"
        # assert vote_weights.shape[0] == 30, "vote_weights must have length 30"
        assert vote_weights.shape[0] == buffer_len, "vote_weights must have same length as buffer_len"
    
    # First element:
    result[0], confidences[0] = arr[0], 1
    
    for i in range(1, total_len):
        if i < buffer_len - 1: # First buffer_len-1 frames
            j = i
            repeat = i + 1 
        elif buffer_len - 1 <= i < total_len - buffer_len + 1: # Middle frames
            j = (i - shift) * buffer_len + shift
            repeat = buffer_len
        elif total_len - buffer_len + 1 <= i: # Last buffer_len-1 frames
            j = (i - shift) * buffer_len + shift
            repeat = total_len - i
        
        sub_arr = arr[j : j + repeat * (buffer_len - 1) : buffer_len - 1]

        assert sub_arr.shape[0] == repeat, f"num_arr: {num_arr}, buffer_len: {buffer_len}, sub_arr shape: {sub_arr.shape}"

        if weighted != "none":
            if sub_arr.shape[0] == vote_weights.shape[0]: 
                sub_arr = np.repeat(sub_arr, vote_weights)


        # Count votes
        values, counts = np.unique(sub_arr, return_counts=True)
        ind = np.argmax(counts)
        elected = values[ind]

        result[i], confidences[i] = elected, counts[ind] / sub_arr.shape[0]

    return result, confidences

def test_majority_voting():
    arrs = []

    T = 5
    N = 3

    for i in range(N):
        arr = np.arange(i, i+T)
        arrs.append(arr)

    arrs = np.concatenate(arrs, 0)

    total_len = N + T - 1

    print(f"shapes: in_len: {arrs.shape}, out_len: {total_len}, buffer_len: {T}")
    res, conf = majority_voting(arrs, total_len, buffer_len=T, weighted="none")

    assert np.all(res == np.arange(total_len)), "Majority voting failed"
    assert np.all(conf==1), "Confidence must be 1"

    print("ok")



def combine_predictions(all_arr, idx, sub_arr, mapping):
    # sub_arr is right hand predictions
    
    batch_size = sub_arr.shape[1]
    assert len(idx) == batch_size, f"2*idx and sub_arr must have same shape: {len(idx)} != {batch_size}"

    right_arr = sub_arr

    for j in range(len(idx)):
        p = right_arr[:, j]
        i = idx[j]

        subject, task, take, start_frame, end_frame = mapping[i]

        # end frame exclusive i.e. [start_frame, end_frame)
        temp_len = end_frame-start_frame

        assert np.all(all_arr[f"{subject}{task}{take}"][temp_len * start_frame : temp_len * (1+start_frame)] == -1), "Not filled appropriately. There should not be any value in all_arr"
        all_arr[f"{subject}{task}{take}"][temp_len * start_frame : temp_len * (1+start_frame)] = p

    return all_arr

def calc_metrics_at_once(all_arr, mapping, num_classes, temporal_length, downsample, weighted_mv, save_dir=None):
    # Applies majority voting, calculates metrics, visualizes and saves the all predictions
    
    # In mapping there is no mirrored data, since it is redundant

    f1_at_k = F1_at_k(num_classes=num_classes)
    f1_classic = MulticlassF1Score(num_classes=num_classes, average='macro')
    f1_micro = MulticlassF1Score(num_classes=num_classes, average='micro')
    conf_mat = MulticlassConfusionMatrix(num_classes=num_classes, normalize="true")

    
    if save_dir is not None:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    for video_id, y_true in mapping.items():
        
        y_pred_all = all_arr[video_id]

        assert -1 not in y_pred_all, "Not filled appropriately. There should not be -1 in y_pred"

        try:
            y_pred, conf = majority_voting(y_pred_all, total_len=y_true[::downsample].shape[0], buffer_len=temporal_length, weighted=weighted_mv)
        except Exception as e:
            raise ValueError(f"Majority voting failed, {video_id}", e)
            
        
        y_pred = y_pred.repeat(downsample)

        pred_len = y_pred.shape[0]
        gt_len = y_true.shape[0]

        if pred_len - gt_len <= downsample:
            y_pred = y_pred[:gt_len]
        elif gt_len - pred_len < downsample:
            y_true = y_true[:pred_len]
        else:
            raise ValueError("gt_len and pred_len must be equal or differ at most by downsample", gt_len, pred_len, video_id)
        
        
        # Notice that the convention is (y_true, y_pred)
        f1_at_k(y_true, y_pred)

        y_pred = y_pred.reshape(*y_true.shape)
        # Notice that the convention is (y_pred, y_true)
        f1_classic(torch.tensor(y_pred), torch.tensor(y_true))

        f1_micro(torch.tensor(y_pred), torch.tensor(y_true))

        # Conf mat
        # To ignore warnings about nan values in confusion matrix and functorch's performance drop
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*NaN values found in confusion matrix.*', UserWarning)

            conf_mat(torch.tensor(y_pred), torch.tensor(y_true))
        
        # Visualization
        if save_dir is not None:
            plot_segmentation_v2(y_true, y_pred, conf, save_file=opj(save_dir, video_id + ".png"), title=video_id)

    # f1_at_10, f1_at_25, f1_at_50, f1_top1_mv, cm_top1_mv
    f1_at_10, f1_at_25, f1_at_50 = f1_at_k.compute()
    ret = {
        "f1_at_10": f1_at_10,
        "f1_at_25": f1_at_25,
        "f1_at_50": f1_at_50,
        "f1_macro_mv": f1_classic.compute().item(),
        "f1_micro_mv": f1_micro.compute().item(),
        "cm_mv": conf_mat.compute().cpu().numpy()
    }
    return ret


def metrics_to_str(metrics):
    selected=["f1_top1", "f1_top3", "f1_top1_once", "f1_micro_once", "f1_top3_once", "f1@10", "f1@25", "f1@50"]
    return " ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k in selected])
    # return " ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if "cm" not in k])

    

class VisIndividualPreds:
    def __init__(self, num_of_samples, training_len, val_len, save_dir) -> None:
        """To visualize individual predictions"""

        # [mode][idx][epoch] -> [y_pred]
        self.segments_pred = {"train": {}, "val": {}}

        # [mode][idx] -> [y_true]
        self.segments_gt = {"train": {}, "val": {}}

        self.num_of_samples = num_of_samples
        self.save_dir = save_dir
        
        self.selected_training_idx=np.random.choice(training_len, num_of_samples, replace=False)
        for s_idx in self.selected_training_idx:
            self.segments_pred["train"][s_idx] = {}


        self.selected_val_idx=np.random.choice(val_len, num_of_samples, replace=False)
        for s_idx in self.selected_val_idx:
            self.segments_pred["val"][s_idx] = {}

    def pick_samples(self, *, y_true, y_pred, idx, epoch, mode):
        # called in each epoch
        # pick samples for visualization
        # save them in self.segments_pred
        # expect y_true and y_pred to be of shape [temp len, batch size]

        assert y_true.shape == y_pred.shape, "y_true and y_pred must have same shape"
        assert len(y_true.shape) == 2, "len(y_true.shape) != 2" 

        selected_idx=self.selected_training_idx if mode=="train" else self.selected_val_idx

        for s_idx in selected_idx:
            # check which selected_idx is in the idx, find mapping and slice it from y_true and y_pred
            if s_idx not in idx:
                continue
            else:
                idx_correspondence = idx.tolist().index(s_idx)

                # For debug purposes, check if there is one and only one assignment.
                if epoch not in self.segments_pred[mode][s_idx]:
                    self.segments_pred[mode][s_idx][epoch] = y_pred[:, idx_correspondence]
                else:
                    raise ValueError("There is more than one assignment")

                if s_idx not in self.segments_gt[mode]:
                    self.segments_gt[mode][s_idx] = y_true[:, idx_correspondence]
                else:
                    assert (self.segments_gt[mode][s_idx] == y_true[:, idx_correspondence]).all(), "Conflicting assignments for ground truth"
                    

    def save(self):
        # call this after loop is finished i.e. epoch == args.max_epochs

        for mode in ["train", "val"]:
            selected_idx=self.selected_training_idx if mode=="train" else self.selected_val_idx

            for s_idx in selected_idx:

                # check if there is an assignment
                plot_segmentation_partial(self.segments_pred[mode][s_idx], self.segments_gt[mode][s_idx], self.save_dir, f"{mode}_{s_idx}")



def print_warning(text, color=(230, 50, 10)):
    print(f"\033[38;2;{color[0]};{color[1]};{color[2]}m{text} \033[38;2;255;255;255m")
 


def get_loss_weights(weighted_loss_type, distribution):

    if weighted_loss_type == "none":
        loss_weights = None
    elif weighted_loss_type == "inv":
        loss_weights = 1.0 / distribution
        loss_weights = loss_weights / loss_weights.sum()
    elif weighted_loss_type == "sqrt_inv":
        loss_weights = 1.0 / torch.sqrt(distribution)
        loss_weights = loss_weights / loss_weights.sum()
    elif weighted_loss_type == "effective":
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, distribution)
        loss_weights = (1.0 - beta) / effective_num
        loss_weights = loss_weights / loss_weights.sum()
    else:
        raise NotImplementedError(f"Weighted loss type {weighted_loss_type} is not implemented")

    return loss_weights


class Metrics:
    def __init__(self, num_classes, temporal_length, downsample, device, dataset_mapping, mode, seg_save_dir, weighted_mv) -> None:
        
        self.dataset_mapping = dataset_mapping
        self.mode = mode
        self.temporal_length = temporal_length
        self.downsample = downsample
        self.seg_save_dir = seg_save_dir 
        self.weighted_mv = weighted_mv 

        # Window-based
        self.f1_top1 = MulticlassF1Score(num_classes=num_classes, top_k=1, average="macro").to(device)
        self.f1_top3 = MulticlassF1Score(num_classes=num_classes, top_k=1, average="macro").to(device) # top_k is intentionally 1

        # These are not really necessary
        if temporal_length == -1:
            self.conf_mat_top1_whole = MulticlassConfusionMatrix(num_classes=num_classes, normalize="true").to(device)
            self.conf_mat_top3_whole = MulticlassConfusionMatrix(num_classes=num_classes, normalize="true").to(device)
        
        # Window-location-sensitive
        if temporal_length != -1:
            self.win_f1_calc_top1 = [MulticlassF1Score(num_classes=num_classes, top_k=1, average="macro").to(device) for _ in range(temporal_length)]
            self.win_f1_calc_top3 = [MulticlassF1Score(num_classes=num_classes, top_k=1, average="macro").to(device) for _ in range(temporal_length)]


        # After majority voting
        # Dictionaries to store predictions for each sample
        # We will also calculate F1 and CM after MV
        if temporal_length != -1:
            self.all_predictions_combined_top1 = {vid: - np.ones((act[::downsample].shape[0] - temporal_length + 1) * temporal_length) for vid, act in dataset_mapping[mode]["vid"].items()}
            self.all_predictions_combined_top3 = {vid: - np.ones((act[::downsample].shape[0] - temporal_length + 1) * temporal_length) for vid, act in dataset_mapping[mode]["vid"].items()}
        else:
            self.f1_at_k = F1_at_k(num_classes=num_classes)


    def update(self, pred, gt_ohe, idxes):
        # gt_ohe: [T, B, C] (one hot encoded)
        # pred: [T, B, C]

        gt = gt_ohe.argmax(dim=-1)
        gt_flattened = gt.flatten(start_dim=0, end_dim=1)

        pred_flattened = pred.flatten(start_dim=0, end_dim=1)

        pred_top_3 = torch.argsort(pred_flattened, dim=-1, descending=True)[:, :3]
        pred_top_1 = pred_top_3[:,0]

        # This magical function ...
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*There is a performance drop.*', UserWarning)
            pred_top_3 = torch.where(torch_vmap(lambda t1, t2: torch.isin(t1, t2))(gt_flattened, pred_top_3), gt_flattened, pred_top_1)
        
        
        # torchmetric
        step_f1_top1 = self.f1_top1(pred_top_1, gt_flattened)
        step_f1_top3 = self.f1_top3(pred_top_3, gt_flattened)

        if self.temporal_length == -1:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', '.*NaN values found in confusion matrix have been replaced with zeros.*', UserWarning)
                self.conf_mat_top1_whole(pred_top_1, gt_flattened)
                self.conf_mat_top3_whole(pred_top_3, gt_flattened)
            # for _ in range(idxes.shape[0]):
            #     self.f1_at_k(, , segment_name="top1")
            


        if self.temporal_length != -1:
            pred_top_1_unf = pred_top_1.unflatten(0, (self.temporal_length, -1))
            pred_top_3_unf = pred_top_3.unflatten(0, (self.temporal_length, -1))

            # Window-location-sensitive metrics
            for i in range(self.temporal_length):
                self.win_f1_calc_top1[i](pred_top_1_unf[i, :], gt[i])
                self.win_f1_calc_top3[i](pred_top_3_unf[i, :], gt[i])

            # These are expects non-flattened tensors
            combine_predictions(self.all_predictions_combined_top1, idxes.numpy(), pred_top_1_unf.cpu(), self.dataset_mapping[self.mode]["sample"])
            combine_predictions(self.all_predictions_combined_top3, idxes.numpy(), pred_top_3_unf.cpu(), self.dataset_mapping[self.mode]["sample"])

        # Only these are matters (actually they don't but whatever)
        return step_f1_top1.cpu().item(), step_f1_top3.cpu().item()


    def compute(self):

        if self.temporal_length != -1:
            # f1_at_10, f1_at_25, f1_at_50, f1_top1_mv, cm_top1_mv
            metrics_top1 = calc_metrics_at_once(self.all_predictions_combined_top1, 
                                                                            self.dataset_mapping[self.mode]["vid"], 
                                                                            num_classes=len(ACTIONS), 
                                                                            temporal_length=self.temporal_length,
                                                                            downsample=self.downsample,
                                                                            weighted_mv=self.weighted_mv,
                                                                            save_dir=self.seg_save_dir)
            # _, _, _, f1_top3_mv, cm_top3_mv
            metrics_top3 = calc_metrics_at_once(self.all_predictions_combined_top3, 
                                                                            self.dataset_mapping[self.mode]["vid"], 
                                                                            num_classes=len(ACTIONS), 
                                                                            temporal_length=self.temporal_length,
                                                                            downsample=self.downsample,
                                                                            weighted_mv=self.weighted_mv,
                                                                            save_dir=None)

            self.win_f1_calc_top1_results = [f1_calc.compute().item() for f1_calc in self.win_f1_calc_top1]
            self.win_f1_calc_top3_results = [f1_calc.compute().item() for f1_calc in self.win_f1_calc_top3]

            ret = {
                "f1_top1": self.f1_top1.compute().item(),
                "f1_top3": self.f1_top3.compute().item(),
                # "cm_top1": self.conf_mat_top1.compute().cpu(),
                # "cm_top3": self.conf_mat_top3.compute().cpu(),
                "f1_top1_once": metrics_top1["f1_macro_mv"],
                "f1_top3_once": metrics_top3["f1_macro_mv"],
                "f1_micro_once": metrics_top1["f1_micro_mv"],
                "cm_top1_once": metrics_top1["cm_mv"],
                "cm_top3_once": metrics_top3["cm_mv"],
                "f1@10": metrics_top1["f1_at_10"],
                "f1@25": metrics_top1["f1_at_25"],
                "f1@50": metrics_top1["f1_at_50"],
                "win_f1_top1": self.win_f1_calc_top1_results,
                "win_f1_top3": self.win_f1_calc_top3_results,
            }
        else:
            ret = {
                "f1_top1_once": self.f1_top1.compute().item(),
                "f1_top3_once": self.f1_top3.compute().item(),
                "cm_top1_once": self.conf_mat_top1_whole.compute().cpu(),
                "cm_top3_once": self.conf_mat_top3_whole.compute().cpu(),
            }
        
        return ret 


def almost_equal(a, b, eps=1e-6):
    return np.abs(a-b) < eps


def wandb_logging(wandb, metric_dict, epoch, mode, whole_mode=False):
    # Gathered all wandb logs here (except batch logs)
    # I think wabdb kind a have state, so I take it as an argument. But not tested otherwise.

    if not whole_mode:
        # Window-location-sensitive metrics: line plots
        lplt1=plot_line(metric_dict["win_f1_top1"], f"{mode}_f1_top1_by_t", do_return_plot=True)
        wandb.log({f"{mode}/f1_by_t": {'top1': wandb.Image(lplt1)}})
        lplt1.close()

        lplt3=plot_line(metric_dict["win_f1_top3"], f"{mode}_f1_top3_by_t", do_return_plot=True)
        wandb.log({f"{mode}/f1_by_t": {'top3': wandb.Image(lplt3)}})
        lplt3.close()

        wandb.log({f"{mode}/epoch": {"f1@10": metric_dict["f1@10"], 
                              "f1@25": metric_dict["f1@25"], 
                              "f1@50": metric_dict["f1@50"], 
                              "f1_top1_once": metric_dict["f1_top1_once"],
                              "f1_micro_once": metric_dict["f1_micro_once"], 
                              "f1_top3_once": metric_dict["f1_top3_once"],
                              "f1_top1": metric_dict["f1_top1"],
                              "f1_top3": metric_dict["f1_top3"]},
                    "epoch": epoch})
    else:
        # TODO: add f1@k
        wandb.log({f"{mode}/epoch": {"f1_top1_once": metric_dict["f1_top1_once"],
                                     "f1_top3_once": metric_dict["f1_top3_once"]},
                   "epoch": epoch})



    # Confusion matrices (only for test and val)
    if mode in ["val", "test"]:
        plt = plot_conf_mat(metric_dict["cm_top1_once"], f"{mode}_CM_top1_e{epoch:0>4}", do_return_plot=True)
        wandb.log({f"{mode}/confusion_matrix": {'top1': wandb.Image(plt)}})
        plt.close()

        plt3 = plot_conf_mat(metric_dict["cm_top3_once"], f"{mode}_CM_top3_e{epoch:0>4}", do_return_plot=True)
        wandb.log({f"{mode}/confusion_matrix": {'top3': wandb.Image(plt3)}})
        plt3.close()

import torch_geometric.nn as g_nn
from torch_geometric.data import Batch

ATTENTION = {}
def set_hooks_for_dist_loss(model):
    def make_hook_fn(name):
        def hook(module, input, output):
            assert isinstance(module, g_nn.TransformerConv)

            edge_indicies, attn = output[1]
            # attn: num_edges x num_heads
            # edge_indicies: 2 x num_edges
            # edge_indicies[:, i]: [source, target] of the i-th edge
            # sum of attentions to target is 1

            ATTENTION[name].append(attn)

            if name == "g_layer_0":
                ATTENTION["edge_indicies"].append(edge_indicies.detach().cpu())

            # Just to be sure
            # if name == "g_layer_1":
                # assert torch.all(edge_indicies.cpu() == ATTENTION["edge_indicies"][])

        return hook

    for i in range(len(model.gnn.gnn) // 3):
        j = 3 * i
        layer_name = f"g_layer_{i}"
        getattr(model.gnn.gnn, f"module_{j}").register_forward_hook(make_hook_fn(layer_name))
        ATTENTION[layer_name] = []
    
    ATTENTION["edge_indicies"] = []


def get_dists_btw_obj(graph_batch: Batch):
    list_of_graphs = graph_batch.to_data_list()
    dists = []
    for graph in list_of_graphs:
        for edge in graph.edge_index.T:
            pos1 = graph.pos[edge[0]]
            pos2 = graph.pos[edge[1]]
            dists.append(torch.dist(pos1, pos2))
            
    dists = torch.tensor(dists)    
    return dists

# def get_attn_dist_loss(graph_list: List[Batch]):
#     # Read attentions.
#     # Calculate the distance between objects.
#     assert len(ATTENTION["edge_indicies"]) == len(ATTENTION["g_layer_0"]) == len(ATTENTION["g_layer_1"]) == 10

#     for graph_batch in graph_list: # temporal
#         list_of_graphs = graph_batch.to_data_list()
#         for graph in list_of_graphs:
#             dists = []
#             for edge_idx, edge in enumerate(graph.edge_index.T):
#                 pos1 = graph.pos[edge[0]]
#                 pos2 = graph.pos[edge[1]]
#                 dists.append(torch.dist(pos1, pos2))
            
#             dists = torch.tensor(dists)
#             graph.dists = dists
#             # graph-wise normalize the distances
        
#         batch = Batch.from_data_list(list_of_graphs)

#         attns = ...
            
#         # define a loss 
            

#     # graphList-> to list of graphs 
#     ...


if __name__ == "__main__":
    test_majority_voting()
