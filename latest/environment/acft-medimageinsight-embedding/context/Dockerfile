# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202605.1

USER root

# Install unzip and upgrade OS packages to fix vulnerabilities
RUN apt-get -y update && apt-get -y upgrade && apt-get -y install unzip && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# mlflow 3.5.0 has CVEs (CVE-2025-14287, CVE-2026-2033, CVE-2026-2635); upgrade after requirements install
# azureml-mlflow pins mlflow-skinny<=3.9.0, so mlflow must be upgraded separately to avoid resolution conflict
RUN pip install --no-cache-dir mlflow==3.11.1
# fastmcp + mcp were installed as regular deps of mlflow 3.5.0 (fastmcp→mcp) but orphaned after
# mlflow 3.11.1 (fastmcp moved to optional "mcp" extra); uninstall both to remove vulnerable packages
RUN pip uninstall -y fastmcp mcp

# Upgrade python-dotenv in the system Python(3.13)
# python-dotenv: transitive dep via pydantic-settings (>=0.21.0 floor); parent uses loose floor so base resolves to
# vulnerable 1.2.1 (GHSA-mf9w-mj56-hr94); override to >=1.2.2
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2'
