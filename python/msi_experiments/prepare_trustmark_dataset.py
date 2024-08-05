import os, sys
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)

from trustmark import TrustMark
from PIL import Image
from pathlib import Path
import math, random, torch
import numpy as np
import pandas as pd
import argparse

def set_random_seeds(seed):
    """
        This function sets all random seed used in this experiment.
        For reproduce purpose.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    # === Some dummt configs ===
    device = torch.device("cuda")
    set_random_seeds(args.random_seed)
    args.watermarker = "TrustMark"

    # === Set the dataset paths ===
    dataset_input_path = os.path.join(
        args.clean_data_root, args.dataset_name
    )
    output_root_path = os.path.join(
        args.clean_data_root, "..", args.watermarker, args.dataset_name
    )
    output_img_root = os.path.join(output_root_path, "encoder_img")
    os.makedirs(output_img_root, exist_ok=True)

    # === Scan all images in the dataset (clean) ===
    img_files = [f for f in os.listdir(dataset_input_path) if ".png" in f]
    print("Total number of images: [{}] --- (Should be 2000)".format(len(img_files)))

    # === Init Trustmark ===
    MODE="C"
    tm = TrustMark(verbose=True, model_type=MODE, use_ECC=False)
    capacity = tm.schemaCapacity()
    bitstring = ''.join([random.choice(['0', '1']) for _ in range(capacity)])
    print("Input Bitstring: ", bitstring)

    # Init the dict to save watermarking summary
    save_csv_dir = os.path.join(output_root_path, "water_mark.csv")
    res_dict = {
        "ImageName": [],
        "Encoder": [],
        "Decoder": [],
        "Match": []
    }

    for img_name in img_files:
        img_clean_path = os.path.join(dataset_input_path, img_name)
        print("***** ***** ***** *****")
        print("Processing Image: {} ...".format(img_clean_path))

        # encoding
        cover = Image.open(img_clean_path)
        width, height = cover.size
        print("Clean Image shape: ({}, {}) ".format(width, height))

        rgb = cover.convert('RGB')
        has_alpha = cover.mode == 'RGBA'
        if (has_alpha):
            alpha=cover.split()[-1]
        encoded = tm.encode(rgb, bitstring, MODE='binary')
        width, height = encoded.size
        print("Watermarked image shape ({}, {}). ".format(width, height))

        img_w_path = os.path.join(output_img_root, img_name)
        encoded.save(img_w_path, exif=cover.info.get('exif'), dpi=cover.info.get('dpi'))
        print("Watermarked img saved to: {}".format(img_w_path))

        # === Sanity Check if watermark is embedded successfully ===
        stego = Image.open(img_w_path).convert('RGB')
        wm_secret, wm_present, wm_schema = tm.decode(stego, MODE='binary')
        print(f'Extracted secret: {wm_secret} (schema {wm_schema})')

        # Compute bitwise acc.
        bit_len = len(bitstring)
        ba = []
        for i in range(bit_len):
            e = bitstring[i]
            d = wm_secret[i]
            if e == d:
                ba.append(1)
            else:
                ba.append(0)
        bitwise_acc = np.mean(ba)
        print("Decode the watermarked image for sanity check (bitwise acc. should be close to 100 %)")
        print("Bitwise acc. {:04f}\%".format(100*bitwise_acc))


        res_dict["ImageName"].append(img_name)
        res_dict["Encoder"].append([bitstring])
        res_dict["Decoder"].append([wm_secret])
        res_dict["Match"].append(bitwise_acc > 0.95)

    df = pd.DataFrame(res_dict)
    df.to_csv(save_csv_dir, index=False)

if __name__ == "__main__":
    print("Use this script to prepare trustmark watermarked dataset.")
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
        "--dataset_name", dest="dataset_name", type=str, help="The dataset name: [COCO, DiffusionDB]",
        default="COCO"
    )
    args = parser.parse_args()
    main(args)
    print("Completed")