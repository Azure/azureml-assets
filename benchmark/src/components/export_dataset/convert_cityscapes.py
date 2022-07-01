"""
This script converts masks from the gtFine CityScapes dataset to convert
labelIds into trainIds, and re-save as PNG files again.

The goal is to allow your image segmentation script to process those masks
without having to implement CityScapes specific logic into your training loop.

To run:
> pip install tqdm numpy pillow
"""
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

# See https://github.com/pytorch/vision/blob/main/torchvision/datasets/cityscapes.py#L67
IGNORE_LABEL = 255
LABEL_TO_TRAIN_ID = {
    -1: IGNORE_LABEL,
    0: IGNORE_LABEL,
    1: IGNORE_LABEL,
    2: IGNORE_LABEL,
    3: IGNORE_LABEL,
    4: IGNORE_LABEL,
    5: IGNORE_LABEL,
    6: IGNORE_LABEL,
    7: 0,
    8: 1,
    9: IGNORE_LABEL,
    10: IGNORE_LABEL,
    11: 2,
    12: 3,
    13: 4,
    14: IGNORE_LABEL,
    15: IGNORE_LABEL,
    16: IGNORE_LABEL,
    17: 5,
    18: IGNORE_LABEL,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: IGNORE_LABEL,
    30: IGNORE_LABEL,
    31: 16,
    32: 17,
    33: 18,
}


def run(args):
    """Run function based on cli args"""
    masks_saved = 0

    for loop_index, mask_path in enumerate(
        Path(args.masks_folder).glob("**/*_gtFine_labelIds.png")
    ):
        if os.path.isfile(mask_path):
            # figure out the path of image relative to masks_folder
            mask_rel_path = os.path.relpath(mask_path, start=args.masks_folder)
            mask_target_path = os.path.join(args.output_folder, mask_rel_path)
            mask_target_folder = os.path.dirname(mask_target_path)

            # load into numpy
            mask_img = Image.open(mask_path)
            mask = np.array(mask_img)
            mask_copy = mask.copy()

            # map labelIds into trainIds
            for label_id, train_id in LABEL_TO_TRAIN_ID.items():
                mask_copy[mask == label_id] = train_id

            # make sure you ignore anything else
            mask_copy[mask > 33] = IGNORE_LABEL
            mask_copy[mask < -1] = IGNORE_LABEL

            # if resulting indices is just IGNORE_LABEL
            mask_copy_indices = np.unique(mask_copy).tolist()

            if mask_copy_indices != [IGNORE_LABEL]:
                # some debug output
                print(
                    f"*** [{masks_saved}/{loop_index}] SAVE {mask_path} => {mask_target_path} ({np.unique(mask)} => {mask_copy_indices})"
                )
                masks_saved += 1
                converted_mask = Image.fromarray(mask_copy.astype(np.uint8))
                os.makedirs(mask_target_folder, exist_ok=True)
                converted_mask.save(mask_target_path)
            else:
                print(
                    f"*** [{masks_saved}/{loop_index}] IGNORED {mask_path} => {mask_target_path} ({np.unique(mask)} => {mask_copy_indices})"
                )
        else:
            print(f"Ignored {mask_path}")


def main(cli_args=None):
    """Parses CLI args and call run()"""
    parser = argparse.ArgumentParser(__doc__)

    parser.add_argument(
        "--output_folder", required=True, type=str, help="where to store masks as PNG"
    )
    parser.add_argument("--masks_folder", required=True, type=str, help="path to gtFine (containiner train, test, val)")

    # actually parses arguments now
    args = parser.parse_args(cli_args)  # if cli_args, uses sys.argv

    run(args)


if __name__ == "__main__":
    main()
