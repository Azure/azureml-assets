# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Model Utils Class."""

import os
import subprocess
import shutil
import stat
import sys
import tempfile
from collections import namedtuple
from pathlib import Path
from typing import Callable, Dict, List
from azureml.assets.util import logger
from azure.ai.ml._azure_environments import (
    AzureEnvironments,
    _get_default_cloud_name,
    _get_storage_endpoint_from_metadata
)

ReplacePair = namedtuple('ReplacePair', ['Original', 'Replacement'])


class CopyUpdater:
    """Update files during azcopy."""

    def __init__(self):
        """Initialize copy updater."""
        self._file_functions: Dict[str, Callable[[Path], bool]] = {}

    def add_file_function(self, file_name: str, func: Callable[[Path], bool]):
        """Add a file function.

        Args:
            file_name (str): Path to file to update, relative to the source path, using POSIX directory separators.
            func (Callable[[Path], bool]): Function that will receive the path to the file to update and returns True
                                           if an update was made, False otherwise.
        """
        self._file_functions[file_name] = func

    @property
    def files(self) -> List[str]:
        """Return the list of files to update."""
        return list(self._file_functions.keys())

    def update_files(self, root_path: Path) -> bool:
        """Update files under the given path.

        Args:
            root_path (Path): Path containing files to update.

        Returns:
            bool: True if any file was updated, False otherwise.
        """
        updated = False
        for file_name, func in self._file_functions.items():
            file_path = root_path / file_name
            if file_path.exists():
                logger.print(f"Updating {file_path}")
                updated |= func(file_path)
            else:
                logger.log_warning(f"{file_path} not found")
        return updated

    @staticmethod
    def create_replace_function(*pairs: ReplacePair) -> Callable[[Path], bool]:
        """Create a function to replace text in a file.

        Args:
            pairs (ReplacePair): Pairs of strings to replace.

        Returns:
            Callable[[Path], bool]: Function that will receive a file and returns whether it was updated.
        """
        def replace_text(file_path: Path) -> bool:
            """Replace text in a file."""
            updated_text = original_text = file_path.read_text()
            for pair in pairs:
                updated_text = updated_text.replace(pair.Original, pair.Replacement)
            if original_text != updated_text:
                file_path.write_text(updated_text)
                return True
            return False

        return replace_text


def _onerror(func, path, exc_info):
    """Error Handler for shutil rmtree."""
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise


def run_cmd(cmd, cwd: Path | None = None, env: dict | None = None) -> int:
    """Run the command and returns the result."""
    logger.print(cmd)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding=sys.stdout.encoding,
        errors="ignore",
        env=env,
    )
    if result.returncode != 0:
        logger.log_error(f"Failed with error {result.stdout}")
    else:
        logger.print(f"Successfully executed! Output: \n{result.stdout}")
    return result.returncode


def download_git_model(model_uri: str, model_dir: Path) -> bool:
    """Download model files from GIT repository.

    Args:
        model_uri (str): git clonable uri of a public repo
        model_dir (Path): local directory to clone model to

    Returns:
        bool: True if successful, False otherwise.
    """
    git = shutil.which("git")
    clone_cmd = [git, "clone", model_uri, model_dir]
    result = run_cmd(clone_cmd)
    if result != 0:
        return False
    git_path = model_dir / ".git"
    shutil.rmtree(git_path, onerror=_onerror)
    return True


def run_azcopy(src_uri: str, dstn_uri: str, include_paths: List[str] = None, exclude_paths: List[str] = None,
               as_subdir: bool = True, overwrite: bool = True, output_level: str = "essential") -> int:
    """Copy blobs between Azure storage accounts or to/from a local dir.

    Args:
        src_uri (str): The source URI.
        dstn_uri (str): The destination URI.
        include_paths (List[str], optional): List of paths to include.
        exclude_paths (List[str], optional): List of paths to exclude.
        as_subdir (bool, optional): If True, copy to a subdirectory under the destination URI.
        overwrite (bool, optional): If True, overwrite the destination blobs.
        output_level (str, optional): Output verbosity level parameter for azcopy. Defaults to "essential".

    Returns:
        int: Return code of azcopy command.
    """
    azcopy = shutil.which("azcopy")
    download_cmd = [azcopy, "copy", src_uri, dstn_uri, "--recursive", "--skip-version-check",
                    "--output-level", output_level]

    # AzureCloud, USGov, and China clouds should all the trusted Microsoft
    # suffixes built into azcop by default. If the cloud is not one of these,
    # then we need to add the appropriate cloud-specific suffix ourselves.
    if _get_default_cloud_name() not in [AzureEnvironments.ENV_DEFAULT,
                                         AzureEnvironments.ENV_US_GOVERNMENT,
                                         AzureEnvironments.ENV_CHINA]:
        suffix = _get_storage_endpoint_from_metadata()
        if not suffix.startswith("."):
            suffix = "." + suffix
        download_cmd.extend(["--trusted-microsoft-suffixes", suffix])

    if not as_subdir:
        download_cmd.append("--as-subdir=false")
    if include_paths:
        download_cmd.extend(["--include-path", ';'.join(include_paths)])
    if exclude_paths:
        download_cmd.extend(["--exclude-path", ';'.join(exclude_paths)])

    if not overwrite:
        download_cmd.append("--overwrite=false")

    # Create a tempdir and set AZCOPY_LOG_LOCATION in env
    with tempfile.TemporaryDirectory() as temp_log_dir:
        env = os.environ.copy()
        env["AZCOPY_LOG_LOCATION"] = temp_log_dir
        result = run_cmd(download_cmd, env=env)
        logger.print(f"azcopy result: {result}")

        if output_level == "default":
            # list all files in the temp log dir
            for root, _, files in os.walk(temp_log_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    logger.print(f"azcopy log file: {file_path}")
                    with open(file_path, "r") as f:
                        logger.print(f.read())

        return result


def copy_azure_artifacts(src_uri: str, dstn_uri: str, copy_updater: CopyUpdater = None,
                         overwrite: bool = True, output_level: str = "essential") -> bool:
    """Copy blobs between Azure storage accounts.

    Args:
        src_uri (str): The source storage account URI.
        dstn_uri (str): The destination storage account URI.
        copy_updater (CopyUpdater): CopyUpdater object to update files during azcopy.
        overwrite (bool): If True, overwrite the destination blobs.
        output_level (str, optional): Output verbosity level parameter for azcopy. Defaults to "essential".

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Copy between storage accounts, excluding any files to be updated
        update_paths = copy_updater.files if copy_updater else None
        result = run_azcopy(src_uri, dstn_uri, exclude_paths=update_paths, overwrite=overwrite,
                            output_level=output_level)
        if result:
            logger.log_error(f"Failed to copy model files from {src_uri}")
            return False

        # Return if no files to update
        if not copy_updater:
            return True

        # Download files to a temporary directory, update, and then upload
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download files to update
            result = run_azcopy(src_uri, temp_dir, include_paths=update_paths, overwrite=overwrite,
                                output_level=output_level)
            if result:
                logger.log_error(f"Failed to download model files to update from {src_uri}")
                return False

            # Update files under the first subdir, created by azcopy by --as-subdir's default of True
            subdir = Path(temp_dir).iterdir().__next__()
            _ = copy_updater.update_files(subdir)

            # Upload updated files
            result = run_azcopy(temp_dir, dstn_uri, as_subdir=False, overwrite=overwrite, output_level=output_level)
            if result:
                logger.log_error(f"Failed to upload updated model files to {dstn_uri}")
                return False

        return True
    except Exception as e:
        logger.log_error(e)
        return False
