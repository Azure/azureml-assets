import json
import os
from pathlib import Path


def debug_output(json_content, output_type, model_name: str, is_local):
    """Output the json content to a file in debug_output folder for local debugging."""
    if not is_local:
        return
    root_dir = os.path.join(os.getcwd(), 'scripts')
    output_dir = Path(root_dir) / "debug_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_debug_{output_type}"
    with open(output_file, "w") as fp:
        json.dump(json_content, fp, indent=2)


# Add highlight in CI logs, according to the https://github.blog/2020-09-23-a-better-logs-experience-with-github-
# actions/#opening-the-door-to-a-more-colorful-experience.
# These key words will be highlighted with colors in the CI logs.
error_logging_prefix = "[Error] "  # Red
warning_logging_prefix = "[Warning] "  # Yellow
debug_logging_prefix = "[Debug] "  # purple


def log(message, prefix="", add_blank_line=False):
    """Log message to the console."""
    if add_blank_line:
        print("")
    print(f"{prefix}{message}")


def log_error(message, add_blank_line=False):
    """Log error message to the console."""
    log(message, error_logging_prefix, add_blank_line)


def log_warning(message, add_blank_line=False):
    """Log warning message to the console."""
    log(message, warning_logging_prefix, add_blank_line)


def log_debug(message, add_blank_line=False):
    """Log debug message to the console."""
    log(message, debug_logging_prefix, add_blank_line)