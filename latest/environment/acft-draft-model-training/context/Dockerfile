#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202605.1
USER root

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# GHSA-jx93-g359-86wm, GHSA-hvwj-8w5g-28rg: sglang vulnerabilities; patched in >=0.5.10
# specforge 0.1.0 pins sglang==0.5.5; override needed after specforge install.
RUN pip install --no-cache-dir --force-reinstall "sglang>=0.5.10"

RUN pip install azureml-acft-common-components==0.0.90
RUN pip install numpy==2.2.5
RUN pip install azureml-evaluate-mlflow==0.0.90

# following are for vulnerability overrides at later
# release of following packages consider moving then to requirements.txt
RUN pip install --no-cache-dir --force-reinstall "mlflow>=3.2.0,<4.0.0"
# wandb>=0.26.0: fixes Go stdlib vulnerabilities (GO-2026-4864/4865/4866/4869/4870/4946/4947)
# in bundled wandb-core binary (Go stdlib v1.26.1 -> v1.26.2)
RUN pip install --no-cache-dir --upgrade "wandb>=0.26.0"
# GHSA-7rgv-gqhr-fxg3: xgrammar stack exhaustion DoS; patched in 0.1.32
RUN pip install xgrammar==0.1.32
# GHSA-69w3-r845-3855 (CVE-2026-1839): arbitrary code execution in Trainer class;
# patched only in transformers>=5.0.0rc3. Upgrading to latest stable 5.x.
RUN pip install transformers==5.5.4
# python-dotenv>=1.2.2: GHSA-mf9w-mj56-hr94; transitive dep in the BASE conda env (python3.13)
#   shipped by the ACPT base image at 1.2.1. Parent packages are pydantic-settings
#   (2.14.0 still requires only python-dotenv>=0.21.0), uvicorn (0.46.0 standard extra
#   requires python-dotenv>=0.13), and fastmcp (3.2.4 requires python-dotenv>=1.1.0) —
#   all use loose floors and no released parent version forces >=1.2.2, so direct override
#   in the base conda env is the only fix path.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2'
# clean conda and pip caches
RUN rm -rf ~/.cache/pip
COPY loss /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/loss.py
COPY eagle3 /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/eagle3.py
COPY parse /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/data/parse.py
