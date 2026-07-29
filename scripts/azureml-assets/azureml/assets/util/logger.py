# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Provide unified logging across GitHub and Azure DevOps, with a fallback to console."""

import os
import threading

_stdout_lock = threading.Lock()
_output_lock = threading.Lock()


class Logger:
    """Base logger class."""

    def log_debug(self, message: str, title: str = None):
        """Log a debug message.

        Args:
            message (str): Message.
            title (str, optional): Message title. Defaults to None.
        """
        pass

    def log_warning(self, message: str, title: str = None):
        """Log a warning message.

        Args:
            message (str): Message.
            title (str, optional): Message title. Defaults to None.
        """
        pass

    def log_error(self, message: str, title: str = None):
        """Log an error message.

        Args:
            message (str): Message.
            title (str, optional): Message title. Defaults to None.
        """
        pass

    def start_group(self, title: str):
        """Start a collapsible group.

        Args:
            title (str): Title.
        """
        pass

    def end_group(self):
        """End a collapsible group."""
        pass

    def set_output(self, name: str, value: str):
        """Set an output variable.

        Args:
            name (str): Name.
            value (str): Value.
        """
        pass

    def print(self, message: str):
        """Print a message, blocking until complete.

        Args:
            message (str): Message.
        """
        with _stdout_lock:
            print(message)


class GitHubLogger(Logger):
    """Logger running under a GitHub agent."""

    def log_debug(self, message: str, title: str = None):
        """Log a debug message.

        Args:
            message (str): Message.
            title (str, optional): Message title. Defaults to None.
        """
        self._log("debug", message, title)

    def log_warning(self, message: str, title: str = None):
        """Log a warning message.

        Args:
            message (str): Message.
            title (str, optional): Message title. Defaults to None.
        """
        self._log("warning", message, title)

    def log_error(self, message: str, title: str = None):
        """Log an error message.

        Args:
            message (str): Message.
            title (str, optional): Message title. Defaults to None.
        """
        self._log("error", message, title)

    def start_group(self, title: str):
        """Start a collapsible group.

        Args:
            title (str): Title.
        """
        self.print(f"::group::{title}")

    def end_group(self):
        """End a collapsible group."""
        self.print("::endgroup::")

    def set_output(self, name: str, value: str):
        """Set an output variable.

        Args:
            name (str): Name.
            value (str): Value.
        """
        output_filename = os.environ["GITHUB_OUTPUT"]
        with _output_lock, open(output_filename, "a", encoding='utf-8') as f:
            f.write(f"{name}={value}\n")

    def _log(self, log_level: str, message: str, title: str = None):
        title_string = f" title={title}" if title is not None else ""
        self.print(f"::{log_level}{title_string}::{message}")


class AzureDevOpsLogger(Logger):
    """Logger running under an Azure DevOps agent."""

    def log_debug(self, message: str, title: str = None):
        """Log a debug message.

        Args:
            message (str): Message.
            title (str, optional): Message title. Defaults to None.
        """
        self._log("debug", message, title)

    def log_warning(self, message: str, title: str = None):
        """Log a warning message.

        Args:
            message (str): Message.
            title (str, optional): Message title. Defaults to None.
        """
        self._log("warning", message, title)

    def log_error(self, message: str, title: str = None):
        """Log an error message.

        Args:
            message (str): Message.
            title (str, optional): Message title. Defaults to None.
        """
        self._log("error", message, title)

    def start_group(self, title: str):
        """Start a collapsible group.

        Args:
            title (str): Title.
        """
        self.print(f"##[group]{title}")

    def end_group(self):
        """End a collapsible group."""
        self.print("##[endgroup]")

    def set_output(self, name: str, value: str):
        """Set an output variable.

        Args:
            name (str): Name.
            value (str): Value.
        """
        self.print(f"##vso[task.setvariable variable={name};isoutput=true]{value}")

    def _log(self, log_level: str, message: str, title: str = None):
        if title is not None:
            title_string = f";title={title}" if title is not None else ""
            self.print(f"##vso[task.logissue type={log_level}{title_string}]{message}")
        else:
            self.print(f"##[{log_level}]{message}")


class ConsoleLogger(Logger):
    """Logger running at the console."""

    def log_debug(self, message: str, title: str = None):
        """Log a debug message.

        Args:
            message (str): Message.
            title (str, optional): Ignored by this logger.
        """
        self._log("debug", message, title)

    def log_warning(self, message: str, title: str = None):
        """Log a warning message.

        Args:
            message (str): Message.
            title (str, optional): Message title.  Ignored by this logger.
        """
        self._log("warning", message, title)

    def log_error(self, message: str, title: str = None):
        """Log an error message.

        Args:
            message (str): Message.
            title (str, optional): Message title. Ignored by this logger.
        """
        self._log("error", message, title)

    def start_group(self, title: str):
        """Ignored by this logger."""
        pass

    def end_group(self):
        """Ignored by this logger."""
        pass

    def set_output(self, name: str, value: str):
        """Ignored by this logger."""
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
