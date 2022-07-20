# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script contains generic methods to handle files and directories.
"""
import os
import glob


def find_image_subfolder(current_root):
    """Identifies the right level of a directory
    that matches with torchvision.datasets.ImageFolder requirements.
    In particular, if images are in current_root/foo/bar/category_X/*.jpg
    we will want to feed current_root/foo/bar/ to ImageFolder.

    Args:
        current_root (str): a given directory

    Returns:
        image_folder (str): the subfolder containing multiple subdirs
    """
    if not os.path.isdir(current_root):
        raise FileNotFoundError(
            f"While identifying the image folder, provided current_root={current_root} is not a directory."
        )

    sub_directories = glob.glob(os.path.join(current_root, "*"))
    if len(sub_directories) == 1:
        # let's do it recursively
        return find_image_subfolder(sub_directories[0])
    if len(sub_directories) == 0:
        raise FileNotFoundError(
            f"While identifying image folder under {current_root}, we found no content at all. The image folder is empty."
        )
    else:
        return current_root
