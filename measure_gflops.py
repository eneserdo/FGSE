import argparse
import glob
import os
import pickle
import random
import time
from os.path import join as opj

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from dataset import get_bimanual_dataset_splits, get_processing_dir
from models import make_model
from utils import arg_parser, seed_worker

# Set up deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')

# For detecting anomalies in gradients
torch.autograd.set_detect_anomaly(True)


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_gflops(model, loader, device, warmup):
    """
    Measure the GFLOPs of the model using PyTorch profiler.
    
    Args:
        model: The PyTorch model to profile
        loader: DataLoader for input data
        device: Device to run the model on
        num_samples: Number of samples to profile
        warmup: Number of warmup iterations
    
    Returns:
        float: Average GFLOPs per forward pass
    """
    model.eval()
    
    # Create CUDA events for timing
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    # Warm up
    print("Warming up...")
    with torch.no_grad():
        for i, unpackable in enumerate(loader):
            if i >= warmup:
                break
            idxes, data = unpackable
            if not isinstance(data, list):
                data = data.to(device)
            else:
                data = [d.to(device) for d in data]
            _ = model(data)
    
    # Synchronize before measurement
    torch.cuda.synchronize()
    
    # Measure with profiler
    num_samples = 0 
    total_flops = 0
    total_time = 0
    
    with torch.no_grad():
        for i, unpackable in enumerate(loader):
            num_samples += 1
            idxes, data = unpackable
            if not isinstance(data, list):
                data = data.to(device)
            else:
                data = [d.to(device) for d in data]
            
            # Start timing
            starter.record()
            
            # Run with profiler to count operations
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_flops=True,
            ) as prof:
                _ = model(data)
            
            # End timing
            ender.record()
            torch.cuda.synchronize()
            
            # Calculate elapsed time in milliseconds
            curr_time = starter.elapsed_time(ender)
            total_time += curr_time
            
            # Extract FLOPs from profiler
            events = prof.key_averages()
            curr_flops = sum(event.flops for event in events if event.flops > 0)
            total_flops += curr_flops
            
            if i % 10 == 0:
                print(f"Sample {i}: {curr_flops / 1e9:.4f} GFLOPs, {curr_time:.2f} ms")
    
    # Calculate averages
    avg_flops = total_flops / num_samples
    avg_time = total_time / num_samples
    
    return avg_flops / 1e9, avg_time  # Convert to GFLOPs



if __name__ == "__main__":
    args = arg_parser()

    args.warmup = 0
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch_generator = torch.Generator()
    torch_generator.manual_seed(args.seed)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataset
    _, test_set, _ = get_bimanual_dataset_splits(
        root=args.root,
        process_it_anyway=args.process_it,
        test_subject=args.test_subject,
        validation_take=args.validation_take,
        temporal_length=args.temporal_length,
        downsample=args.downsample,
        use_vf=args.use_vf,
        filtered_data=args.filtered_data,
        monograph=args.monograph
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_set, 
        args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=True, 
        worker_init_fn=seed_worker, 
        generator=torch_generator
    )
    
    # Create model
    model = make_model(args).to(device)
    
    # Find the best model in the saved directory
    save_dir = opj("saved_models", f"{args.name}")
    weight_list = sorted(glob.glob(opj(save_dir, "*.pt")))
    
    if weight_list:
        best_model_path = weight_list[-1]  # Use the latest model by default
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model restored from {best_model_path}")
    else:
        print("No saved model found. Using randomly initialized weights.")
    
    # Print model information
    print(model)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Get dataset mapping
    processed_dir = get_processing_dir(
        args.root, 
        args.temporal_length, 
        args.downsample, 
        args.use_vf, 
        args.filtered_data, 
        args.monograph
    )
    dataset_mapping_dir = opj(processed_dir, f"dataset_mapping_fold{args.test_subject}.pickle")
    with open(dataset_mapping_dir, "rb") as f:
        dataset_mapping = pickle.load(f)
    
    # Measure GFLOPs
    gflops, avg_time = measure_gflops(
        model, 
        test_loader, 
        device, 
        warmup=args.warmup
    )
    
    print(f"\nResults:")
    print(f"Average GFLOPs: {gflops:.4f}")
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"FPS: {1000 / avg_time:.2f}") 