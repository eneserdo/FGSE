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
from dataset import (ACTIONS, _get_max_length, get_bimanual_dataset_splits,
                     get_processing_dir)
from models import make_model
from utils import (Metrics, VisIndividualPreds, arg_parser, get_loss_weights,
                   get_minutes, metrics_to_str, seed_worker,
                   visualize_incorrect_preds, wandb_logging)

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


def run_one_epoch(model: torch.nn.Module, loader: DataLoader, mode: str, epoch: int):

    if mode in ["test", "val"]:
        model.eval()
    elif mode == "train":
        model.train()

    loss_cumulative = 0

    seg_save_dir = opj(save_dir, f"test_results_{epoch}") if (mode == "test" and args.save_segments) else None
    metrics = Metrics(len(ACTIONS), args.temporal_length, args.downsample, device, dataset_mapping, mode, seg_save_dir, args.weighted_mv)
    

    for step, unpackable in enumerate(loader):
        idxes, data = unpackable

        if not args.monograph:
            data = [d.to(device) for d in data]
            gt_reordered = torch.stack([torch.cat([d.y[::2], d.y[1::2]]) for d in data], dim=0)
        else:
            batch_size = data.y.shape[0] // args.temporal_length
            data = data.to(device)
            gt_reordered = torch.cat([data.y[:,0,:].reshape(batch_size, args.temporal_length, len(ACTIONS)).permute(1, 0, 2), 
                                      data.y[:,1,:].reshape(batch_size, args.temporal_length, len(ACTIONS)).permute(1, 0, 2)], dim=1)

        # gt_reordered: [temporal_length, 2*batch_size(left, ..., left, right, ..., right) , num_classes]

        with torch.cuda.amp.autocast(enabled=not args.no_amp):
            pred = model(data)

            pred_flattened = pred.flatten(0, 1)
            gt_flattened = gt_reordered.flatten(0, 1)
            
            loss = loss_criterion(pred_flattened, gt_flattened)
            loss = loss / args.num_acc

        
        loss_cumulative += loss.item()
        
        if mode == "train":
            # loss.backward()
            scaler.scale(loss).backward()
            
            # optimizer.step()
            if (step + 1) % args.num_acc == 0 or step == len(loader) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        


        if mode != "test" and args.save_individual:
            # Visualize individiual predictions
            raise NotImplementedError("save_individual may be broken due to global idx, please check it")
            VIP.pick_samples(y_true=gt_reordered.cpu().argmax(2), y_pred=pred.detach().cpu(), idx=idxes, epoch=epoch, mode=mode)


        step_f1_top1, step_f1_top3 = metrics.update(pred.detach(), gt_reordered, idxes)


        if mode != "test":
            wandb.log({f"{mode}/batch": {'top1_F1': step_f1_top1, 'top3_F1': step_f1_top3, "Loss": loss.item()}, 'batch': step})

        if mode == "test" and args.save_incorrects:
            raise NotImplementedError("save_incorrects may be broken, please check it")
            # visualize_incorrect_preds(pred_top_1, ground_truth, idxes, num_of_samples=4, test_idx_sample_mapping=dataset_mapping["test"]["sample"])
        
        # if (step+1)%20==0:
        #     print("20 BATCH ÇALIŞIYOR SADECE")
        #     break

    # End of for loop
    
    metric_dict = metrics.compute()

    loss_cumulative = loss_cumulative / (len(loader.dataset) / args.batch_size)



    return metric_dict, loss_cumulative


def test_for_all_saved_weights(model, test_loader, choose_best_in_terms_of):
    # Log everything for cross-validation 

    cv_file = opj(save_dir, f"cross_validation_fold{args.test_subject}.pickle")
    cv_list = []

    comparison_metric=choose_best_in_terms_of
    
    weight_list=sorted(glob.glob(opj(save_dir, "*.pt")))

    if len(weight_list) == 0:
        return None
    
    best_metric = 0    
    results = []
    best_i = -1
    for i, w_name in enumerate(weight_list):
        print(f"TESTING {w_name}:")
        checkpoint = torch.load(w_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        returned_dict, _ = run_one_epoch(model, test_loader, 'test', epoch=int((w_name[-7:-3])))
        cv_list.append(returned_dict)
        
        print('### TEST: ', metrics_to_str(returned_dict))

        results.append(returned_dict)        
        if best_metric < returned_dict[comparison_metric]:
            best_metric = returned_dict[comparison_metric]
            best_i = i

    with open(cv_file, "wb") as f:
        pickle.dump(cv_list, f)

    print('### BEST: ', metrics_to_str(results[best_i]))
    return results[best_i]


if __name__ == "__main__":
    args = arg_parser()
    print("ARGS:\n", args)

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
            

    wandb.init(project="rt-action-segmentation-from-scene-graphs", name=args.name, config=args.__dict__,
               mode="disabled" if args.disable_wandb else "online",
               tags=tags)


    train_set, test_set, val_set = get_bimanual_dataset_splits(root=args.root, 
                                                               process_it_anyway=args.process_it,
                                                               test_subject=args.test_subject,
                                                               validation_take=args.validation_take,
                                                               temporal_length=args.temporal_length,
                                                               downsample=args.downsample,
                                                               use_vf=args.use_vf,
                                                               filtered_data=args.filtered_data,
                                                               monograph=args.monograph)

    train_loader = DataLoader(train_set, args.batch_size, num_workers=args.num_workers, shuffle=True, worker_init_fn=seed_worker, generator=torch_generator)
    
    test_loader = DataLoader(test_set, args.batch_size, num_workers=args.num_workers, shuffle=True, worker_init_fn=seed_worker, generator=torch_generator)

    val_loader = DataLoader(val_set, args.batch_size, num_workers=args.num_workers, shuffle=True, worker_init_fn=seed_worker, generator=torch_generator)

    training_len = len(train_set)
    val_len = len(val_set)

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
    wandb.log({"total_params": total_params, "rnn_params": rnn_params, "gnn_params": gnn_params})


    wandb.watch(model, log='all')
    wandb.define_metric("epoch")
    wandb.define_metric("train/epoch*", step_metric="epoch")
    wandb.define_metric("val/epoch*", step_metric="epoch")
    wandb.define_metric("test/epoch*", step_metric="epoch")
    # wandb.define_metric("test*", step_metric="epoch")
    # wandb.define_metric("val/epoch/*", step_metric="epoch", summary="max")
    wandb.define_metric("val/confusion_matrix*", step_metric="epoch")
    wandb.define_metric("test/confusion_matrix*", step_metric="epoch")
    wandb.define_metric("batch")
    wandb.define_metric("train/batch*", step_metric="batch")
    wandb.define_metric("val/batch*", step_metric="batch")
    # wandb.define_metric("test_top1")
    # wandb.define_metric("test_top3")



    if args.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = getattr(torch.optim, args.optimizer_type)(model.parameters(), lr=args.learning_rate,
                                                              weight_decay=args.weight_decay)

    if args.scheduler_step_size != -1:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size,
                                                    gamma=args.scheduler_gamma, verbose=True)
    
    else:
        # TODO: bunu loss a bakmak yerine f1 a bakarak yapmak daha mantıklı olabilir
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_gamma, patience=4, verbose=True)


    scaler = torch.cuda.amp.GradScaler()

    # loss weights can be used to balance the classes using different strategies
    loss_weights = get_loss_weights(args.weighted_loss, train_set.distribution)

    loss_criterion = torch.nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=args.smoothing).to(device=device)


    save_dir = opj("saved_models", f"{args.name}")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)


    # Load saved models if required
    if args.restore is not None:
        assert args.restored_name is not None, "restored_name must be provided if restore is not None"
        checkpoint = torch.load(opj("saved_models", f"{args.restored_name}", f"model_epoch{args.restore:0>4}.pt"))

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Model is restored successfully from epoch {args.restore}!")
        
        rprint("[magenta]Optimizer and scheduler are overwritten with the restored ones")

    
    # dataset_mapping contains mapping for each sample to its location in the videos. 
    # Also, a mapping from video to ground truth action labels
    processed_dir = get_processing_dir(args.root, args.temporal_length, args.downsample, args.use_vf, args.filtered_data, args.monograph)
    dataset_mapping_dir = opj(processed_dir, f"dataset_mapping_fold{args.test_subject}.pickle")
    with open(dataset_mapping_dir, "rb") as f:
        dataset_mapping = pickle.load(f)
        
    if args.save_individual:
        VIP = VisIndividualPreds(20, training_len, val_len, save_dir)

    if args.temporal_length == -1:
        MAX_VID_LEN = _get_max_length(args.root, args.downsample)

    # STARTS TRAINING AND VALIDATION
    print("Starts training...")
    start = 1 if args.restore is None else args.restore
    epoch = start
    for epoch in range(start, start + args.max_epochs):
        start = time.perf_counter()

        train_metrics, _ = run_one_epoch(model, train_loader, 'train', epoch)
        wandb_logging(wandb, train_metrics, epoch, "train", whole_mode=args.temporal_length == -1)
        
        val_metrics, val_loss = run_one_epoch(model, val_loader, 'val', epoch)
        wandb_logging(wandb, val_metrics, epoch, "val", whole_mode=args.temporal_length == -1)

        # Save model
        if epoch % args.saving_freq == 0 or epoch == start + args.max_epochs - 1:
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict()},
                        opj(save_dir, f"model_epoch{epoch:0>4}.pt"))

        if args.scheduler_step_size == -1:
            scheduler.step(val_loss)
        else:
            scheduler.step()
        stop = time.perf_counter()

        print(f'### {epoch:03d}: Time {get_minutes(stop - start)} m ### Train: ', metrics_to_str(train_metrics), ' ### Val: ', metrics_to_str(val_metrics), flush=True)


    # save the individual predictions
    if args.save_individual:
        VIP.save()

    # TEST
    print("Starts test...")

    choose_best_in_terms_of = "f1_top1_once"
    test_metrics = test_for_all_saved_weights(model, test_loader, choose_best_in_terms_of=choose_best_in_terms_of)
    
    if test_metrics is None:
        rprint("[red] No weights found to test")
    else:
        wandb_logging(wandb, test_metrics, epoch, "test", whole_mode=args.temporal_length == -1)
    
    wandb.finish()
    
    try:
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e):
            time.sleep(5)
        else:
            raise e