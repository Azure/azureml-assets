# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import threading

_stdout_lock = threading.Lock()
_output_lock = threading.Lock()

class Logger:
    def log_debug(self, message: str, title: str = None):
        pass

    def log_warning(self, message: str, title: str = None):
        pass

    def log_error(self, message: str, title: str = None):
        pass

    def start_group(self, title: str):
        pass

    def end_group(self):
        pass

    def set_output(self, name: str, value: str):
        pass

    def print(self, message: str):
        with _stdout_lock:
            print(message)


class GitHubLogger(Logger):
    def log_debug(self, message: str, title: str = None):
        self._log("debug", message, title)

    def log_warning(self, message: str, title: str = None):
        self._log("warning", message, title)

    def log_error(self, message: str, title: str = None):
        self._log("error", message, title)

    def start_group(self, title: str):
        self.print(f"::group::{title}")

    def end_group(self):
        self.print("::endgroup::")

    def set_output(self, name: str, value: str):
        output_filename = os.environ["GITHUB_OUTPUT"]
        with _output_lock, open(output_filename, "a") as f:
            f.write(f"{name}={value}\n")

    def _log(self, log_level: str, message: str, title: str = None):
        title_string = f" title={title}" if title is not None else ""
        self.print(f"::{log_level}{title_string}::{message}")


class AzureDevOpsLogger(Logger):
    def log_debug(self, message: str, title: str = None):
        self._log("debug", message, title)

    def log_warning(self, message: str, title: str = None):
        self._log("warning", message, title)

    def log_error(self, message: str, title: str = None):
        self._log("error", message, title)

    def start_group(self, title: str):
        self.print(f"##[group]{title}")

    def end_group(self):
        self.print("##[endgroup]")

    def set_output(self, name: str, value: str):
        self.print(f"##vso[task.setvariable variable={name};isoutput=true]{value}")

    def _log(self, log_level: str, message: str, title: str = None):
        if title is not None:
            title_string = f";title={title}" if title is not None else ""
            self.print(f"##vso[task.logissue type={log_level}{title_string}]{message}")
        else:
            self.print(f"##[{log_level}]{message}")


class ConsoleLogger(Logger):
    def log_debug(self, message: str, title: str = None):
        self._log("debug", message, title)

    def log_warning(self, message: str, title: str = None):
        self._log("warning", message, title)

    def log_error(self, message: str, title: str = None):
        self._log("error", message, title)

    def start_group(self, title: str):
        pass

    def end_group(self):
        pass

    def set_output(self, name: str, value: str):
        pass

    def _log(self, log_level: str, message: str, title: str = None):
        self.print(f"{log_level}: {message}")


def _create_default_logger() -> Logger:
    if os.environ.get("GITHUB_RUN_NUMBER") is not None:
        return GitHubLogger()
    elif os.environ.get("BUILD_BUILDNUMBER") is not None:
        return AzureDevOpsLogger()
    else:
        return ConsoleLogger()


# Create default logger
logger = _create_default_logger()
