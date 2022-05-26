import tempfile
from pathlib import Path

import azureml.assets as assets
import azureml.assets.environment as environment

RESOURCES_DIR = Path("resources/environment")


def test_build_assets(test_subdir: str, expected: bool, resource_group: str, registry: str):
    this_dir = Path(__file__).parent.parent

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        assert environment.build_images(
            input_dirs=this_dir / RESOURCES_DIR / test_subdir,
            asset_config_filename=assets.DEFAULT_ASSET_FILENAME,
            output_directory=temp_dir_path / "output",
            build_logs_dir=temp_dir_path / "build_logs",
            pin_versions=True,
            max_parallel=5,
            changed_files=[],
            tag_with_version=False,
            resource_group=resource_group,
            registry=registry,
            test_command="python -V",
            ) == expected
