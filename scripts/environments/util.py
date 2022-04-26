import filecmp
import os
import shutil

from config import AssetConfig

OUTPUT_SUBDIR_TEMPLATE = "{type}/{name}"
RELEASE_SUBDIR = "latest"


# See https://stackoverflow.com/questions/4187564/recursively-compare-two-directories-to-ensure-they-have-the-same-files-and-subdi
def are_dir_trees_equal(dir1: str, dir2: str) -> bool:
    """Compare two directories recursively based on files names and content.

    Args:
        dir1 (str): First directory
        dir2 (str): Second directory

    Returns:
        bool: True if the directory trees are the same and there were no errors
            while accessing the directories or files, False otherwise.
    """

    dirs_cmp = filecmp.dircmp(dir1, dir2)
    if dirs_cmp.left_only or dirs_cmp.right_only or dirs_cmp.funny_files:
        return False
    (_, mismatch, errors) = filecmp.cmpfiles(dir1, dir2, dirs_cmp.common_files, shallow=False)
    if mismatch or errors:
        return False
    for common_dir in dirs_cmp.common_dirs:
        new_dir1 = os.path.join(dir1, common_dir)
        new_dir2 = os.path.join(dir2, common_dir)
        if not are_dir_trees_equal(new_dir1, new_dir2):
            return False
    return True


def copy_replace_dir(source: str, dest: str):
    """Copy a directory tree, replacing any existing one.

    Args:
        source (str): Source directory
        dest (str): Destination directory
    """
    # Delete destination directory
    if os.path.exists(dest):
        shutil.rmtree(dest)

    # Copy source to destination directory
    shutil.copytree(source, dest)


def get_asset_output_dir(asset_config: AssetConfig, output_directory_root: str) -> str:
    """Generate the output directory for a given asset.

    Args:
        asset_config (AssetConfig): Asset config
        output_directory_root (str): Output directory root

    Returns:
        str: The output directory
    """
    output_subdir = OUTPUT_SUBDIR_TEMPLATE.format(type=asset_config.type.value,
                                                  name=asset_config.name)
    return os.path.join(output_directory_root, output_subdir)


def get_asset_release_dir(asset_config: AssetConfig, release_directory_root: str) -> str:
    """Generate the release directory for a given asset.

    Args:
        asset_config (AssetConfig): Asset config
        release_directory_root (str): Release directory root

    Returns:
        str: The release directory
    """
    return get_asset_output_dir(asset_config, os.path.join(release_directory_root, RELEASE_SUBDIR))


def copy_asset_to_output_dir(asset_config: AssetConfig, output_directory_root: str):
    """Copy asset directory to output directory.

    Args:
        asset_config (AssetConfig): Asset config to copy
        output_directory_root (str): Output directory root
    """
    output_directory = get_asset_output_dir(asset_config, output_directory_root)
    copy_replace_dir(asset_config.file_path, output_directory)
