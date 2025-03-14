import argparse
import os
import pathlib

from dataset import (create_relation_files, download_coax_dataset,
                     get_processing_dir, process_coax_dataset)


def arg_parser_dataset():
    ap = argparse.ArgumentParser()

    # Dataset
    ap.add_argument("-rt", "--root", type=str, default="./coax", help="Root directory of dataset.")
    
    ap.add_argument("-d", "--download_anyway", action="store_true", help="download dataset")
    ap.add_argument("--unzip_anyway", action="store_true", help="unzip dataset")
    
    ap.add_argument("-tl", "--temporal_length", type=int, required=True, help="Temporal length of the input (if -1, processes whole video)")
    ap.add_argument("-ds", "--downsample", type=int, default=1, help="Downsampling ratio")
    
    ap.add_argument("--filtered_data", action="store_true", help="Filter")
    
    return ap.parse_args()


def main():
    print("Preparing dataset...")
    args = arg_parser_dataset()

    if args.downsample <= 0 and int(args.downsample) != args.downsample:
        raise ValueError("It must be an integer greater than 0")

    # check if it is already ready
    pr_dir = pathlib.Path(get_processing_dir(args.root, args.temporal_length, args.downsample, False, args.filtered_data, False))

    if pr_dir.exists():
        raise ValueError(f"The dataset is already processed in {pr_dir}\n, (Remove it manually if necessary)")


    if args.unzip_anyway:
        if not os.path.exists(args.root):
            os.makedirs(args.root)
        download_coax_dataset(args.root, args.download_anyway)
        create_relation_files(args.root)


    process_coax_dataset(args.root, args.temporal_length, args.downsample, False, args.filtered_data)
    print("Dataset processed!")
    print("You can now train your model!")


if __name__ == "__main__":
    main()
