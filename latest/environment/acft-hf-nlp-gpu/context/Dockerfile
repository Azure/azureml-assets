#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py38-torch201:biweekly.202401.2

USER root

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip

COPY requirements.txt .
# The below 2 files are required for baking the code into the environment
COPY data_import_run.py /azureml/data_import/run.py
COPY finetune_run.py /azureml/finetune/run.py

RUN pip install "azureml-evaluate-mlflow==0.0.42"
# pyarrow-hotfix is installed to fix the vulnerability CVE-2023-47248
# This can be removed once datasets is upgraded to >= 2.14.7
# mlflow==2.9.2 and mlflow-skinny==2.9.2 are installed to fix the vulnerabilities
# GHSA-cxfr-5q3r-2rc2, GHSA-v945-r3rc-6fjm, GHSA-vwhf-3v6x-wff8 and GHSA-wqxf-447m-6f5f
# This can be removed once azureml-evaluate-mlflow updates the dependency.
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt
RUN MAX_JOBS=4 pip install flash-attn==2.3.3 --no-build-isolation

# dummy number to change when needing to force rebuild without changing the definition: 1
