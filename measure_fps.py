import glob
import os
import pickle
import random
import time
from os.path import join as opj

import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import torch
from rich import print as rprint
from torch_geometric.loader import DataLoader

import wandb
from dataset import (
    ACTIONS,
    _get_max_length,
    get_bimanual_dataset_splits,
    get_processing_dir,
)
from models import make_model
from utils import (
    Metrics,
    VisIndividualPreds,
    arg_parser,
    get_loss_weights,
    get_minutes,
    metrics_to_str,
    seed_worker,
    visualize_incorrect_preds,
    wandb_logging,
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# # To fix the following error:
# OSError: [Errno 24] Too many open files
# Alternatively ulimit can be increased
torch.multiprocessing.set_sharing_strategy('file_system')

# # An attempt to fix the following error:
# XIO:  fatal IO error 25 (Inappropriate ioctl for device) on X server "localhost:10.0" after 2627 requests (2627 known processed) with 8 events remaining.
pyplot.ioff()
matplotlib.use('Agg')

# To check for NaNs in the gradients
torch.autograd.set_detect_anomaly(True)


def run_for_fps(model: torch.nn.Module, loader: DataLoader, count: int = 100, warmup: int = 5):

    model.eval()
    
    i = 0

    for step, unpackable in enumerate(loader):
        idxes, data = unpackable
        if i == warmup:
            stime = time.perf_counter()
        
        if i == count + warmup:
            break

        data = [d.to(device) for d in data]
        # gt_reordered = torch.stack([torch.cat([d.y[::2], d.y[1::2]]) for d in data], dim=0)

        # gt_reordered: [temporal_length, 2*batch_size(left, ..., left, right, ..., right) , num_classes]
        
        with torch.cuda.amp.autocast(enabled=not args.no_amp):
            pred = model(data)
            # pred_flattened = pred.flatten(0, 1)
            # gt_flattened = gt_reordered.flatten(0, 1)       
        i += 1
    etime = time.perf_counter()

    return 1/((etime - stime)/count)


if __name__ == "__main__":
    args = arg_parser()
    #### asserts about args:

    if args.temporal_length ==-1:
        assert args.norm_layer == "LayerNorm", "When temporal_length is -1, norm_layer must be LayerNorm, otherwise it may break the norm stats"
        assert args.temporal_type == "tr", "others are not implemented yet"

    if args.use_vf:
        assert not args.use_pos and not args.use_embedding, "use_vf cannot be used with use_pos or use_embedding"
        assert args.use_global_pooling, "use_vf must use global pooling"

    if args.temporal_length == -1:
        if args.batch_size != 1:
            args.num_acc *= args.batch_size
            args.batch_size = 1
            print(f"args.batch_size is set to 1, and args.num_acc is set to {args.num_acc} for whole video processing")

    assert args.batch_size == 1, "batch_size must be 1 for fps measurement"

    # Make sure it is reproducible
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.use_deterministic_algorithms(mode=True, warn_only=True) # This throws an error
    torch_generator = torch.Generator()
    torch_generator.manual_seed(args.seed)

    # assert args.gpu_id < torch.cuda.device_count(), f"gpu_id:{args.gpu_id} not found. Total gpu count: {torch.cuda.device_count()}"
    device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    tags = []
    td_tag = "T" + str(args.temporal_length) if args.temporal_length != -1 else "W"
    td_tag += "d" + str(args.downsample)
    tags.append(td_tag)

    if hasattr(args, "sweep"):
        if args.sweep:
            sweep_tag = "sweep"
            tags.append(sweep_tag)
            


    train_set, test_set, val_set = get_bimanual_dataset_splits(root=args.root, 
                                                               process_it_anyway=args.process_it,
                                                               test_subject=args.test_subject,
                                                               validation_take=args.validation_take,
                                                               temporal_length=args.temporal_length,
                                                               downsample=args.downsample,
                                                               use_vf=args.use_vf,
                                                               filtered_data=args.filtered_data,
                                                               monograph=args.monograph)


    test_loader = DataLoader(test_set, args.batch_size, num_workers=args.num_workers, shuffle=True, worker_init_fn=seed_worker, generator=torch_generator)

    model = make_model(args).to(device)

    # if True:
    #     checkpoint = torch.load(opj(f"saved_Beta1_10d3_SG_fold{args.test_subject}", "model_epoch0016.pt"))
    #     # model.gnn.load_state_dict(torch.load(args.pretrained_gnn))
    #     gnn_net = checkpoint["model_state_dict"]
    #     gnn_net.pop("temporal_layer.weight")
    #     gnn_net.pop("temporal_layer.bias")
    #     model.load_state_dict(gnn_net, strict=False)


    print(model)
    if args.monograph:
        rnn_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    else:
        rnn_params = sum(p.numel() for p in model.temporal_layer.parameters() if p.requires_grad)
    gnn_params = sum(p.numel() for p in model.gnn.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total trainable params: {total_params:_}")
    print(f"RNN: {rnn_params:_}")
    print(f"GNN: {gnn_params:_}")



    # Load saved models if required
    if args.restore is not None:
        assert args.restored_name is not None, "restored_name must be provided if restore is not None"
        checkpoint = torch.load(opj(f"saved_{args.restored_name}", f"model_epoch{args.restore:0>4}.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model is restored successfully from epoch {args.restore}!")

    
    # dataset_mapping contains mapping for each sample to its location in the videos. 
    # Also, a mapping from video to ground truth action labels
    processed_dir = get_processing_dir(args.root, args.temporal_length, args.downsample, args.use_vf, args.filtered_data, args.monograph)
    dataset_mapping_dir = opj(processed_dir, f"dataset_mapping_fold{args.test_subject}.pickle")
    with open(dataset_mapping_dir, "rb") as f:
        dataset_mapping = pickle.load(f)
        
    # TEST
    print("Starts test...")
    with torch.no_grad():
        fps = run_for_fps(model, test_loader, count=100, warmup=5)
        print(f"Average FPS: {fps:.2f}")


    try:
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e):
            time.sleep(5)
        else:
            raise e
