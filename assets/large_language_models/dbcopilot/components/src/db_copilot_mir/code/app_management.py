# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""App management module."""
import importlib
import json
import logging
from dataclasses import dataclass

from flask import Flask


@dataclass
class AppConfig:
    """App config."""

    module_name: str = "app"
    app_name: str = "app"
    port: int = 8080

    @staticmethod
    def load_from_file(file: str = "app_config.json"):
        """Load the app config from file."""
        with open(file) as f:
            data = json.load(f)
        return AppConfig(**data)


def start_flask_app(config: AppConfig):
    """Start the flask app."""
    logging.info(f"Starting Flask app: {config.module_name}.{config.app_name} at port {config.port}")
    app_module = importlib.import_module(f"app_code.{config.module_name}")
    app = getattr(app_module, config.app_name)
    assert isinstance(app, Flask)
    app.run(host="localhost", port=config.port)