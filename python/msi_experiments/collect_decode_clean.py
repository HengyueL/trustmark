"""
    This script is use dwtDctSvd/rivaGan to decode clean images to compute FPR.
"""
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)
import os, argparse, torch
import random
import numpy as np
import pandas as pd
from general import rgb2bgr, save_image_bgr, set_random_seeds, \
    watermark_np_to_str, watermark_str_to_numpy
from trustmark import TrustMark
from PIL import Image


def main(args):
    # === Some dummt configs ===
    device = torch.device("cuda")
    set_random_seeds(args.random_seed)

    # === Set the dataset paths ===
    dataset_input_path = os.path.join(
        args.clean_data_root, args.dataset
    )
    # === Scan all images in the dataset (clean) ===
    img_files = [f for f in os.listdir(dataset_input_path) if ".png" in f]
    print("Total number of images: [{}] --- (Should be 100)".format(len(img_files)))

    output_root_path = os.path.join(
        "..", "..", "DIP_Watermark_Evasion", 
        "dataset", "Clean_Watermark_Evasion", args.watermarker, args.dataset
    )
    os.makedirs(output_root_path, exist_ok=True)
    # === Init watermarker ===
    MODE="C"
    tm = TrustMark(verbose=True, model_type=MODE, use_ECC=False)
    capacity = tm.schemaCapacity()
    
    # Init the dict to save watermarking summary
    save_csv_dir = os.path.join(output_root_path, "water_mark.csv")
    res_dict = {
        "ImageName": [],
        "Decoder": [],
    }

    for img_name in img_files:
        img_clean_path = os.path.join(dataset_input_path, img_name)
        print("***** ***** ***** *****")
        print("Processing Image: {} ...".format(img_clean_path))
        stego = Image.open(img_clean_path).convert('RGB')
        wm_secret, wm_present, wm_schema = tm.decode(stego, MODE='binary')
        watermark_decode_str = wm_secret

        res_dict["ImageName"].append(img_name)
        res_dict["Decoder"].append([watermark_decode_str])

    df = pd.DataFrame(res_dict)
    df.to_csv(save_csv_dir, index=False)


if __name__ == "__main__":
    print("Use this script to download DiffusionDB dataset.")
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    parser.add_argument(
        "--random_seed", dest="random_seed", type=int, help="Manually set random seed for reproduction.",
        default=13
    )
    parser.add_argument(
        '--clean_data_root', type=str, help="Root dir where the clean image dataset is located.",
        default=os.path.join("..", "..", "DIP_Watermark_Evasion", "dataset", "Clean")
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str, help="The dataset name: [COCO, DiffusionDB]",
        default="COCO"
    )
    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, help="Specification of watermarking method. ['dwtDctSvd', 'rivaGan']",
        default="TrustMark"
    )
    args = parser.parse_args()
    main(args)
    print("Completd")