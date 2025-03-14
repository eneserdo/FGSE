import argparse
import os
import pathlib

from dataset import (download_bimanual_dataset, get_processing_dir,
                     process_bimanual_dataset)


def arg_parser_dataset():
    ap = argparse.ArgumentParser()

    # Dataset
    ap.add_argument("-rt", "--root", type=str, default=".", help="Root directory of dataset.")
    
    ap.add_argument("-d", "--download_anyway", action="store_true", help="download dataset")
    ap.add_argument("--unzip_anyway", action="store_true", help="unzip dataset")
    
    ap.add_argument("--fix_fps", action="store_true", help="Fix fps of the videos to 30 fps")
    ap.add_argument("--fix_null", action="store_true", help="Fix null actions in the video of subject_6-task_4-take_1")
    
    ap.add_argument("-tl", "--temporal_length", type=int, required=True, help="Temporal length of the input (if -1, processes whole video)")
    ap.add_argument("-ds", "--downsample", type=int, default=1, help="Downsampling ratio")

    ap.add_argument("-vf", "--use_vf", action="store_true", help="Use visual features")
    
    ap.add_argument("--filtered_data", action="store_true", help="Filter")
    
    return ap.parse_args()


def main():
    print("Preparing dataset...")
    args = arg_parser_dataset()

    if args.downsample <= 0 and int(args.downsample) != args.downsample:
        raise ValueError("It must be an integer greater than 0")

    # check if it is already ready
    pr_dir = pathlib.Path(get_processing_dir(args.root, args.temporal_length, args.downsample, args.use_vf, args.filtered_data, False))

    if pr_dir.exists():
        raise ValueError(f"The dataset is already processed in {pr_dir}\n, (Remove it manually if necessary)")


    raw_dir = os.path.join(args.root, "raw")

    if args.unzip_anyway:
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
        download_bimanual_dataset(raw_dir, args.fix_null, args.fix_fps, args.download_anyway)
        

    process_bimanual_dataset(args.root, args.temporal_length, args.downsample, args.use_vf, args.filtered_data)
    print("Dataset processed!")
    print("You can now train your model!")


if __name__ == "__main__":
    main()
