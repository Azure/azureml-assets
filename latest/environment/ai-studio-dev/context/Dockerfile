FROM mcr.microsoft.com/devcontainers/python:3.10

WORKDIR /

# needed for build_mlindex
RUN apt update && apt upgrade -y && apt install -y fuse

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Adding this install under root's path for the custom AI Studio container solution
ENV PATH="/home/vscode/.dotnet/tools:${PATH}"

# The instructions run after this for AI CLI are required to be run by the vscode user
WORKDIR /home/vscode
USER vscode

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Install Azure Developer CLI (AZD)
RUN curl -fsSL https://aka.ms/install-azd.sh | bash

# The VS Code dev container base image contains both Python 3.10 and 3.11, with 3.11 taking priority.
# The following is a workaround to set the default Python interpreter to 3.10. This is currently a requirement for the AI CLI, SDK and Prompt flow packages.
RUN mkdir -p /home/vscode/.vscode-server/data/Machine
RUN mkdir -p /home/vscode/.vscode-server-insiders/data/Machine
RUN echo '{ "python.defaultInterpreterPath": "/usr/local/bin/python", "workbench.editorAssociations": { "**/afh/**/README.md": "vscode.markdown.preview.editor" } }' > /home/vscode/.vscode-server/data/Machine/settings.json
RUN echo '{ "python.defaultInterpreterPath": "/usr/local/bin/python", "workbench.editorAssociations": { "**/afh/**/README.md": "vscode.markdown.preview.editor" } }' > /home/vscode/.vscode-server-insiders/data/Machine/settings.json

# Temporary fix for the security issue: remove the old version of package setuptools from the Python container
RUN FILE="/usr/local/lib/python3.10/ensurepip/_bundled/setuptools-65.5.0-py3-none-any.whl"; if [ -f "$FILE" ]; then sudo rm -f "$FILE"; fi
RUN FILE="/opt/az/lib/python3.11/ensurepip/_bundled/setuptools-65.5.0-py3-none-any.whl"; if [ -f "$FILE" ]; then sudo rm -f "$FILE"; fi
RUN DIR="/usr/local/py-utils/shared/lib/python3.10/site-packages/setuptools-65.5.0.dist-info"; if [ -d "$DIR" ]; then sudo rm -rf "$DIR"; fi
RUN DIR="/usr/local/py-utils/venvs/black"; if [ -d "$DIR" ]; then sudo rm -rf "$DIR"; fi
