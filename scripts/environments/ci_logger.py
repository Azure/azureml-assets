import os


class Logger:
    def log_debug(self, message, title=None):
        pass

    def log_warning(self, message, title=None):
        pass

    def log_error(self, message, title=None):
        pass

    def start_group(self, title):
        pass

    def end_group(self):
        pass


class GitHubLogger(Logger):
    def log_debug(self, message, title=None):
        self._log("debug", message, title)

    def log_warning(self, message, title=None):
        self._log("warning", message, title)

    def log_error(self, message, title=None):
        self._log("error", message, title)

    def start_group(self, title):
        print(f"::group::{title}")

    def end_group(self):
        print("::endgroup::")

    def _log(self, log_level, message, title=None):
        title_string = f" title={title}" if title is not None else ""
        print(f"::{log_level}{title_string}::{message}")


class AzureDevOpsLogger(Logger):
    def log_debug(self, message, title=None):
        self._log("debug", message, title)

    def log_warning(self, message, title=None):
        self._log("warning", message, title)

    def log_error(self, message, title=None):
        self._log("error", message, title)

    def start_group(self, title):
        print(f"##[group]{title}")

    def end_group(self):
        print("##[endgroup]")

    def _log(self, log_level, message, title=None):
        if title is not None:
            title_string = f";title={title}" if title is not None else ""
            print(f'##vso[task.logissue type={log_level}{title_string}]{message}')
        else:
            print(f'##[{log_level}]{message}')


class ConsoleLogger(Logger):
    def log_debug(self, message, title=None):
        self._log("debug", message, title)

    def log_warning(self, message, title=None):
        self._log("warning", message, title)

    def log_error(self, message, title=None):
        self._log("error", message, title)

    def start_group(self, title):
        pass

    def end_group(self):
        pass

    def _log(self, log_level, message, title=None):
        print(f"{log_level}: {message}")


def _create_default_logger() -> Logger:
    if os.environ.get("GITHUB_RUN_NUMBER") is not None:
        return GitHubLogger()
    elif os.environ.get("BUILD_BUILDNUMBER") is not None:
        return AzureDevOpsLogger()
    else:
        return ConsoleLogger()


# Create default logger
logger = _create_default_logger()
