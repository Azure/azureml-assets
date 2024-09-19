# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Script utility methods."""

import difflib
import filecmp
import os
import re
import shutil
from pathlib import Path, PurePath
from ruamel.yaml import YAML
from typing import List, Tuple, Union

import azureml.assets as assets
from azureml.assets.util import logger
from azureml.assets.config import ValidationException

RELEASE_SUBDIR = "latest"
EXCLUDE_DIR_PREFIX = "!"


# See https://stackoverflow.com/questions/4187564
def are_dir_trees_equal(dir1: Path, dir2: Path, enable_logging: bool = False, ignore_eol: bool = True) -> bool:
    """Compare two directories recursively based on files names and content.

    Args:
        dir1 (Path): First directory
        dir2 (Path): Second directory
        enable_logging (bool, optional): Enable logging for mismatches
        ignore_eol (bool, optional): Ignore EOL differences when comparing files

    Returns:
        bool: True if the directory trees are the same and there were no errors
            while accessing the directories or files, False otherwise.
    """
    dirs_cmp = filecmp.dircmp(dir1, dir2)
    if dirs_cmp.left_only:
        _log_diff(f"Compared {dir1} and {dir2} and found these only in {dir1}: {dirs_cmp.left_only}", enable_logging)
        return False
    if dirs_cmp.right_only:
        _log_diff(f"Compared {dir1} and {dir2} and found these only in {dir2}: {dirs_cmp.right_only}", enable_logging)
        return False
    if dirs_cmp.funny_files:
        _log_diff(f"Compared {dir1} and {dir2} and couldn't compare these: {dirs_cmp.funny_files}", enable_logging)
        return False
    (_, differences, errors) = filecmp.cmpfiles(dir1, dir2, dirs_cmp.common_files, shallow=False)
    if differences:
        _log_diff(f"Compared {dir1} and {dir2} and found differences in: {differences}", enable_logging)
        non_eol_differences = []
        for file in differences:
            if _are_files_equal_ignore_eol(dir1 / file, dir2 / file):
                _log_diff(f"Ignoring differences for {file} because they're only related to EOLs", enable_logging)
            else:
                _log_file_diff(dir1 / file, dir2 / file, enable_logging)
                non_eol_differences.append(file)
        if non_eol_differences:
            return False
    if errors:
        _log_diff(f"Compared {dir1} and {dir2} and couldn't compare these: {errors}", enable_logging)
        return False
    for common_dir in dirs_cmp.common_dirs:
        new_dir1 = dir1 / common_dir
        new_dir2 = dir2 / common_dir
        if not are_dir_trees_equal(new_dir1, new_dir2, enable_logging):
            return False
    return True


def _log_diff(message: str, enabled: bool):
    if enabled:
        logger.log_warning(message)


def _log_file_diff(file1: Path, file2: Path, enabled: bool):
    if enabled:
        with open(file1, "r") as file1_obj, open(file2, "r") as file2_obj:
            file1_contents = file1_obj.readlines()
            file2_contents = file2_obj.readlines()
            diff = difflib.unified_diff(file1_contents, file2_contents, str(file1), str(file2))
            logger.print("".join(diff))


def _are_files_equal_ignore_eol(file1: Path, file2: Path) -> bool:
    with open(file1, "r") as file1_obj, open(file2, "r") as file2_obj:
        while True:
            line1 = file1_obj.readline()
            line2 = file2_obj.readline()
            line1 = line1.rstrip("\n\r") if line1 else None
            line2 = line2.rstrip("\n\r") if line2 else None
            if line1 != line2:
                return False
            if line1 is None and line2 is None:
                return True


def _resolve_from_file(value):
    if os.path.isfile(value):
        with open(value, 'r') as f:
            content = f.read()
            return (True, content)
    else:
        return (False, None)


def resolve_from_file_for_asset(asset: assets.AssetConfig, value):
    """Resolve the value from a file for an asset if it is a file, otherwise returns the value.

    Args:
        asset (AssetConfig): the asset to try and resolve the value for
        value: value to try and resolve
    """
    if not is_file_relative_to_asset_path(asset, value):
        return value

    path_value = value if isinstance(value, Path) else Path(value)

    if not path_value.is_relative_to(asset.file_path):
        path_value = asset._append_to_file_path(path_value)

    (is_resolved_from_file, resolved_value) = _resolve_from_file(path_value)

    if is_resolved_from_file:
        return resolved_value
    else:
        return value


def is_file_relative_to_asset_path(asset: assets.AssetConfig, value):
    """Check if the value from is a file with respect to the asset path.

    Args:
        asset (AssetConfig): the asset to try and resolve the value for
        value: value to check
    """
    if not isinstance(value, str) and not isinstance(value, PurePath):
        return False

    path_value = value if isinstance(value, Path) else Path(value)

    if not path_value.is_relative_to(asset.file_path):
        path_value = asset._append_to_file_path(path_value)

    return os.path.isfile(path_value)


def copy_replace_dir(source: Path, dest: Path, paths: List[Path] = None):
    """Copy a directory tree, replacing any existing one.

    Args:
        source (Path): Source directory
        dest (Path): Destination directory
        paths (List[Paths], optional): Specific paths to copy, relative to the source directory
    """
    # Delete destination directory
    if dest.exists():
        shutil.rmtree(dest)

    # Copy source to destination directory
    if not paths:
        # Easy, copy everything
        shutil.copytree(source, dest)
    else:
        # Copy only selected paths
        for path in paths:
            source_path = source / path
            dest_path = dest / path
            if source_path.is_dir():
                Path.mkdir(dest_path, parents=True, exist_ok=True)
            else:
                Path.mkdir(dest_path.parent, parents=True, exist_ok=True)
                shutil.copyfile(source_path, dest_path)


def get_asset_output_dir(asset_config: assets.AssetConfig, output_directory_root: Path,
                         use_version_dir: bool = False) -> Path:
    """Generate the output directory for a given asset.

    Args:
        asset_config (assets.AssetConfig): Asset config
        output_directory_root (Path): Output directory root
        use_version_dir (bool, optional): Add version-specific directory

    Returns:
        Path: The output directory
    """
    version = asset_config.version if use_version_dir else None
    return get_asset_output_dir_from_parts(asset_config.type, asset_config.name, output_directory_root, version)


def get_asset_output_dir_from_parts(type: assets.AssetType, name: str, output_directory_root: Path,
                                    version: str = None) -> Path:
    """Generate the output directory for a given asset.

    Args:
        type (assets.AssetConfig): Asset type
        name (str): Asset name
        output_directory_root (Path): Output directory root
        version (str, optional): Asset version

    Returns:
        Path: The output directory
    """
    output_directory = Path(output_directory_root, type.value, name)
    if version is not None:
        output_directory = output_directory / version

    return output_directory


def get_asset_release_dir(asset_config: assets.AssetConfig, release_directory_root: Path) -> Path:
    """Generate the release directory for a given asset.

    Args:
        asset_config (assets.AssetConfig): Asset config
        release_directory_root (Path): Release directory root

    Returns:
        Path: The release directory
    """
    return get_asset_output_dir(asset_config, release_directory_root / RELEASE_SUBDIR)


def get_asset_release_dir_from_parts(type: assets.AssetType, name: str, release_directory_root: Path) -> Path:
    """Generate the release directory for a given asset.

    Args:
        type (assets.AssetConfig): Asset type
        name (str): Asset name
        release_directory_root (Path): Release directory root

    Returns:
        Path: The release directory
    """
    return get_asset_output_dir_from_parts(type, name, release_directory_root / RELEASE_SUBDIR)


def copy_asset_to_output_dir(asset_config: assets.AssetConfig, output_directory: Path, add_subdir: bool = False,
                             use_version_dir: bool = False) -> Path:
    """Copy asset directory to output directory.

    Args:
        asset_config (assets.AssetConfig): Asset config to copy
        output_directory_root (Path): Output directory root
        add_subdir (bool, optional): Add asset-specific subdirectories to output_directory
        use_version_dir (bool, optional): Store asset in version-specific directory

    Returns:
        Path: The asset's output directory
    """
    if add_subdir:
        output_directory = get_asset_output_dir(asset_config, output_directory, use_version_dir)
    elif use_version_dir:
        output_directory = output_directory / asset_config.version

    common_dir, relative_release_paths = find_common_directory(asset_config.release_paths)
    copy_replace_dir(source=common_dir, dest=output_directory, paths=relative_release_paths)
    return output_directory


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
    assert len(components) == 2
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


def _convert_excludes(input_dirs: Union[List[Path], Path],
                      exclude_dirs: List[Path] = None) -> Tuple[List[Path], List[Path]]:
    """Extract directories to exclude from input_dirs and add them to exclude_dirs.

    Args:
        input_dirs (Union[List[Path], Path]): Directories to search in.
        exclude_dirs (List[Path], optional): Directories that should be excluded from the search.

    Returns:
        Tuple[List[Path], List[Path]]: _description_
    """
    if type(input_dirs) is not list:
        input_dirs = [input_dirs]
    if exclude_dirs is not None:
        if type(exclude_dirs) is not list:
            exclude_dirs = [exclude_dirs]
    else:
        exclude_dirs = []

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

    return input_dirs, exclude_dirs


def find_assets(input_dirs: Union[List[Path], Path],
                asset_config_filename: str = assets.DEFAULT_ASSET_FILENAME,
                types: Union[List[assets.AssetType], assets.AssetType] = None,
                changed_files: List[Path] = None,
                exclude_dirs: List[Path] = None,
                pattern: re.Pattern = None) -> List[assets.AssetConfig]:
    """Search directories for assets.

    Args:
        input_dirs (Union[List[Path], Path]): Directories to search in.
        asset_config_filename (str, optional): Asset config filename to search for.
        types (Union[List[assets.AssetType], assets.AssetType], optional): AssetTypes to search for.
        changed_files (List[Path], optional): Changed files, used to filter assets in input_dirs.
        exclude_dirs (Union[List[Path], Path], optional): Directories that should be excluded from the search.
        pattern (re.Pattern, optional): Regex pattern used to filter assets. Defaults to None.

    Returns:
        List[assets.AssetConfig]: Assets found.
    """
    if types is not None and type(types) is not list:
        types = [types]

    # Find and filter assets
    found_assets = []
    for asset_config_file in find_asset_config_files(input_dirs, asset_config_filename, changed_files, exclude_dirs):
        asset_config = assets.AssetConfig(asset_config_file)

        # If specified, skip types not included in filter
        if types and asset_config.type not in types:
            continue

        # If specified, skip assets that don't match the pattern
        if pattern is not None and not pattern.fullmatch(asset_config.full_name):
            continue

        found_assets.append(asset_config)
    return found_assets


def find_asset_config_files(input_dirs: Union[List[Path], Path],
                            asset_config_filename: str,
                            changed_files: List[Path] = None,
                            exclude_dirs: List[Path] = None) -> List[Path]:
    """Search directories for asset config files.

    Args:
        input_dirs (Union[List[Path], Path]): Directories to search in.
        asset_config_filename (str): Asset config filename to search for.
        changed_files (List[Path], optional): Changed files, used to filter assets in input_dirs.
        exclude_dirs (Union[List[Path], Path], optional): Directories that should be excluded from the search.

    Returns:
        List[Path]: Asset config files found.
    """
    input_dirs, exclude_dirs = _convert_excludes(input_dirs, exclude_dirs)
    changed_files_resolved = set([file.resolve() for file in changed_files] if changed_files else [])

    found_assets = []
    for input_dir in input_dirs:
        for file in input_dir.rglob(asset_config_filename):
            # If specified, skip assets when no change in asset, source_code and test_code
            try:
                asset_config = assets.AssetConfig(file)
                test_dir_path = asset_config.pytest_tests_dir_with_path
                test_dir_path = test_dir_path.resolve() if test_dir_path else Path()

                release_paths_resolved = [path.resolve() for path in asset_config.release_paths]
                asset_changed_files = changed_files_resolved & set(release_paths_resolved)
            except ValidationException:
                test_dir_path = Path()
                asset_changed_files = set()

            if changed_files and not asset_changed_files and not any(
                file.parent in f.parents or
                test_dir_path in f.resolve().parents
                for f in changed_files
            ):
                continue

            # If specified, skip excluded directories
            if exclude_dirs and any([d for d in exclude_dirs if d in file.parents]):
                continue

            found_assets.append(file)
    return found_assets


def find_files(input_dirs: Union[List[Path], Path],
               filename: str) -> List[Path]:
    """Search directories for files.

    Args:
        input_dirs (Union[List[Path], Path]): Directories to search in.
        filename (str): Filename to search for.

    Returns:
        List[Path]: Files found (excludes directories).
    """
    found_files = []
    for input_dir in input_dirs:
        for file in input_dir.rglob(filename):
            if file.is_file():
                found_files.append(file)

    return found_files


def find_common_directory(paths: List[Path]) -> Tuple[Path, List[Path]]:
    """Find lowest common directory for a list of Paths.

    Args:
        paths (List[Path]): Paths to consider.

    Returns:
        Tuple[Path, List[Path]]: Common directory and updated Paths which are relative to it.
    """
    lowest_common_dirs = None
    paths_resolved = [p.resolve() for p in paths]
    for path in paths_resolved:
        # Create list of dirs
        path_resolved_parents = list(path.parents)
        if path.is_dir():
            dirs = [path]
            dirs.extend(path_resolved_parents)
        else:
            dirs = path_resolved_parents
        dirs.reverse()

        if lowest_common_dirs is None:
            # Starting point
            lowest_common_dirs = dirs
        else:
            # Find and store common directory
            min_len = min([len(lowest_common_dirs), len(dirs)])
            for i in range(min_len - 1, 0, -1):
                if lowest_common_dirs[i].samefile(dirs[i]):
                    lowest_common_dirs = lowest_common_dirs[0:i + 1]
                    break
            else:
                raise Exception(f"Unable to find a common path between {lowest_common_dirs[-1]} and {dirs[-1]}")

    # Assemble output
    lowest_common_dir = lowest_common_dirs[-1]
    relative_paths = [p.relative_to(lowest_common_dir) for p in paths_resolved]

    return lowest_common_dir, relative_paths


def load_yaml(file_path: str) -> dict:
    """Load yaml utility.

    Args:
        file_path (str): yaml file path

    Returns:
        dict: loaded yaml as dict
    """
    with open(file_path, "r") as f:
        yaml_dict = YAML().load(f)
    return yaml_dict


def dump_yaml(yaml_dict: dict, file_path: str):
    """Dump yaml utility.

    Args:
        yaml_dict (str): dictionary object to dump into yaml.
        file_path (str): File path to store dump result to.
    """
    with open(file_path, "w") as f:
        yaml_dict = YAML().dump(yaml_dict, f)


def retry(times):
    """Retry Decorator.

    Args:
        times (int): The number of times to repeat the wrapped function/method
    """

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 1
            while attempt <= times:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    attempt += 1
                    ex_msg = "Exception thrown when attempting to run {}, attempt {} of {}".format(
                        func.__name__, attempt, times
                    )
                    logger.log_warning(ex_msg)
                    if attempt == times:
                        logger.log_warning(
                            "Retried {} times when calling {}, now giving up!".format(times, func.__name__)
                        )
                        raise

        return newfn

    return decorator
