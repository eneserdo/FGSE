# Gets all arguments and start subprocesses using those arguments
# collect all outputs and calculate mean and std for each metric
# choose best one and log the mean and std to wandb

import os
import pickle
import subprocess
import time
from os.path import join as opj

import numpy as np

import wandb
from utils import arg_parser_cv as arg_parser

METRICS_OF_INTEREST = ["f1_top1_once", "f1_micro_once", "f1_top3_once", "f1@10", "f1@25", "f1@50"]
COMPARISON_METRIC = "f1_top1_once"

def main():
    args = arg_parser()

    args_str = ""
    for key, value in args.__dict__.items():
        if key in ["name", "num_folds", "num_sweeps", "test_subject"]:
            continue
        
        if value is None:
            continue

        if isinstance(value, bool):
            if value:
                args_str += f" --{key}"
        
        elif isinstance(value, list):
            if len(value) == 0:
                continue
            args_str += f" --{key} {' '.join(map(str, value))}"

        else:
            args_str += f" --{key} {value}"


    for i in range(1, 1+args.num_folds):
        print("#######################\n",
              "####### FOLD", i, "########\n",
              "#######################\n")

        saved_dir = opj("saved_models", f"{args.name}_fold{i}")
        cv_file = opj(saved_dir, f"cross_validation_fold{i}.pickle")

        if os.path.exists(cv_file):
            print(f"Fold {i} already exists. Skipping...")
            continue
        
        subject_arg = f" --test_subject {i}"
        name_arg = f"{args.name}_fold{i}"

        ret = subprocess.call("python main.py " + args_str + subject_arg + f" --name {name_arg}", shell=True)
        if ret != 0:
            print("Error occured. Can't continue, exiting cv...")
            exit(1)
        time.sleep(5)


    all_outputs = [] # list of list of dictionaries


    for i in range(1, 1+args.num_folds):
        saved_dir = opj("saved_models", f"{args.name}_fold{i}")
        cv_file = opj(saved_dir, f"cross_validation_fold{i}.pickle")

        with open(cv_file, "rb") as f:
            data = pickle.load(f)
            
        all_outputs.append(data)

    num_weights = len(data)

    processed_outputs = []

    for f in range(args.num_folds):
        temp_arr = np.array([all_outputs[f][e][COMPARISON_METRIC] for e in range(num_weights)])
        best_e = np.argmax(temp_arr)

        processed_outputs.append(all_outputs[f][best_e])

    metrics = {}

    for m in METRICS_OF_INTEREST:
        temp_arr = np.array([processed_outputs[f][m] for f in range(args.num_folds)])
        metrics[m] = {}
        metrics[m]["mean"] = np.mean(temp_arr)
        metrics[m]["std"] = np.std(temp_arr)


    # final_metrics = []

    # for _ in range(num_weights):
    #     processed_outputs.append({ key: [] for key in METRICS_OF_INTEREST })
    #     final_metrics.append({ key: {} for key in METRICS_OF_INTEREST })

    # for e in range(num_weights):
    #     for k in METRICS_OF_INTEREST:
    #         for i in range(args.num_folds):
    #             processed_outputs[e][k].append(all_outputs[i][e][k])
            
    #         arr = np.array(processed_outputs[e][k])
    #         final_metrics[e][k]["mean"] = np.mean(arr)
    #         final_metrics[e][k]["std"] = np.std(arr)
    
    if args.num_folds > 1:
        cv_n = f"CV{args.num_folds}"
        args.name = cv_n + "_" + args.name

        wandb.init(project="rt-action-segmentation-from-scene-graphs", 
                name=args.name, 
                config=args.__dict__,
                mode="disabled" if args.disable_wandb else "online",
                tags=[cv_n])
        

        # best_e=0
        # best_score=0
        # for e in range(num_weights):
        #     if best_score < final_metrics[e][COMPARISON_METRIC]["mean"]:
        #         best_score = final_metrics[e][COMPARISON_METRIC]["mean"]
        #         best_e = e

        # metrics = final_metrics[best_e]
        for key in metrics.keys():
            mean = metrics[key]["mean"]
            std = metrics[key]["std"]

            # I preferred them to appear under the same column in wandb 
            # even though they have different number of folds 
            cv_key = "CV_" + key
            print(f"{cv_key} mean: {mean}, std: {std}")
            wandb.log({f"{cv_key}_mean": mean, f"{cv_key}_std": std})

            if key == COMPARISON_METRIC:            
                # wandb.log({f"{key}_f{f+1}": processed_outputs[best_e][key][f] for f in range(args.num_folds)})
                wandb.log({f"{key}_f{f+1}": processed_outputs[f][key] for f in range(args.num_folds)})

        wandb.finish()


if __name__ == "__main__":
    main()