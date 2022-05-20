import filecmp
import shutil
from pathlib import Path
from typing import List, Union

import azureml.assets as assets

RELEASE_SUBDIR = "latest"
EXCLUDE_DIR_PREFIX = "!"


# See https://stackoverflow.com/questions/4187564/recursively-compare-two-directories-to-ensure-they-have-the-same-files-and-subdi
def are_dir_trees_equal(dir1: Path, dir2: Path) -> bool:
    """Compare two directories recursively based on files names and content.

    Args:
        dir1 (Path): First directory
        dir2 (Path): Second directory

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
        new_dir1 = dir1 / common_dir
        new_dir2 = dir2 / common_dir
        if not are_dir_trees_equal(new_dir1, new_dir2):
            return False
    return True


def copy_replace_dir(source: Path, dest: Path):
    """Copy a directory tree, replacing any existing one.

    Args:
        source (Path): Source directory
        dest (Path): Destination directory
    """
    # Delete destination directory
    if dest.exists():
        shutil.rmtree(dest)

    # Copy source to destination directory
    shutil.copytree(source, dest)


def get_asset_output_dir(asset_config: assets.AssetConfig, output_directory_root: Path) -> Path:
    """Generate the output directory for a given asset.

    Args:
        asset_config (assets.AssetConfig): Asset config
        output_directory_root (Path): Output directory root

    Returns:
        Path: The output directory
    """
    return Path(output_directory_root, asset_config.type.value, asset_config.name)


def get_asset_release_dir(asset_config: assets.AssetConfig, release_directory_root: Path) -> Path:
    """Generate the release directory for a given asset.

    Args:
        asset_config (assets.AssetConfig): Asset config
        release_directory_root (Path): Release directory root

    Returns:
        Path: The release directory
    """
    return get_asset_output_dir(asset_config, release_directory_root / RELEASE_SUBDIR)


def copy_asset_to_output_dir(asset_config: assets.AssetConfig, output_directory_root: Path):
    """Copy asset directory to output directory.

    Args:
        asset_config (assets.AssetConfig): Asset config to copy
        output_directory_root (Path): Output directory root
    """
    output_directory = get_asset_output_dir(asset_config, output_directory_root)
    copy_replace_dir(asset_config.file_path, output_directory)


def apply_tag_template(full_image_name: str, template: str = None) -> str:
    """Apply a template to an image's tag.

    Args:
        full_image_name (str): The full image name, which must include a tag suffix.
        template (str): Template to use. If desired, use {tag} as a placeholder for the existing tag value.

    Returns:
        str: The transformed image name.
    """
    if template is None:
        return full_image_name

    components = full_image_name.rsplit(":", 1)
    components[-1] = template.format(tag=components[-1])
    return ":".join(components)


def apply_version_template(version: str, template: str = None) -> str:
    """Apply a template to a version string.

    Args:
        version (str): The version.
        template (str): Template to use. If desired, use {version} as a placeholder for the existing version string.

    Returns:
        str: The transformed version.
    """
    if template is None:
        return version
    return template.format(version=version)


def find_assets(input_dirs: Union[List[Path], Path],
                asset_config_filename: str,
                types: Union[List[assets.AssetType], assets.AssetType] = None,
                changed_files: List[Path] = None,
                exclude_dirs: List[Path] = None) -> List[assets.AssetConfig]:
    """Search directories for assets.

    Args:
        input_dirs (Union[List[Path], Path]): Directories to search in.
        asset_config_filename (str): Asset config filename to search for.
        types (Union[List[assets.AssetType], assets.AssetType], optional): AssetTypes to search for. Will not filter if unspecified.
        changed_files (List[Path], optional): Changed files, used to filter assets in input_dirs. Will not filter if unspecified.
        exclude_dirs (Union[List[Path], Path], optional): Directories that should be excluded from the search.

    Returns:
        List[assets.AssetConfig]: Assets found.
    """
    if type(input_dirs) is not list:
        input_dirs = [input_dirs]
    if types is not None and type(types) is not list:
        types = [types]
    if exclude_dirs and type(exclude_dirs) is not list:
        exclude_dirs = [exclude_dirs]

    # Exclude any dirs that start with EXCLUDE_DIR_PREFIX
    new_input_dirs = []
    new_exclude_dirs = []
    for input_dir in input_dirs:
        input_dir_str = str(input_dir)
        if input_dir_str.startswith(EXCLUDE_DIR_PREFIX):
            new_exclude_dirs.append(Path(input_dir_str[len(EXCLUDE_DIR_PREFIX):]))
        else:
            new_input_dirs.append(input_dir)
    if new_exclude_dirs:
        input_dirs = new_input_dirs
        if exclude_dirs:
            exclude_dirs.extend(new_exclude_dirs)
        else:
            exclude_dirs = new_exclude_dirs

    # Find and filter assets
    found_assets = []
    for asset_config_file in find_asset_config_files(input_dirs, asset_config_filename):
        asset_config = assets.AssetConfig(asset_config_file)

        # If specified, skip types not included in filter
        if types and asset_config.type not in types:
            continue

        # If specified, skip assets with no changed files
        if changed_files and not any([f for f in changed_files if asset_config.file_path in f.parents]):
            continue

        # If specified, skip excluded directories
        if exclude_dirs and any([d for d in exclude_dirs if d in asset_config.file_path.parents]):
            continue

        found_assets.append(asset_config)
    return found_assets


def find_asset_config_files(input_dirs: Union[List[Path], Path],
                            asset_config_filename: str) -> List[Path]:
    """Search directories for asset config files.

    Args:
        input_dirs (Union[List[Path], Path]): Directories to search in.
        asset_config_filename (str): Asset config filename to search for.

    Returns:
        List[Path]: Asset config files found.
    """
    if type(input_dirs) is not list:
        input_dirs = [input_dirs]

    found_assets = []
    for input_dir in input_dirs:
        found_assets.extend(input_dir.rglob(asset_config_filename))
    return found_assets
