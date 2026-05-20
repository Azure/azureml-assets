#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}
USER root

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# GHSA-jx93-g359-86wm, GHSA-hvwj-8w5g-28rg: sglang vulnerabilities; patched in >=0.5.10.
# specforge 0.1.0 pins sglang==0.5.5. Reinstall only the top-level wheel to avoid pulling
# newer torch/flashinfer stacks that bloat the image and break compatibility in this env.
RUN pip install --no-cache-dir --force-reinstall --no-deps "sglang==0.5.10"

RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install numpy==2.2.5
RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}

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
# pyOpenSSL>=26.0.0 fixes the DTLS cookie callback overflow/bypass issues pulled in via
# azureml-core; urllib3/cryptography floors keep the transitive SSL stack on patched builds.
RUN pip install --no-cache-dir --upgrade 'pyOpenSSL>=26.0.0' 'urllib3>=2.7.0' 'cryptography>=46.0.7'
# Base conda env (python3.13) carries vulnerable python-dotenv/urllib3/pyOpenSSL/idna builds.
# Parent packages use loose floors, so direct overrides are required until the base image refreshes.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2' 'pyOpenSSL>=26.0.0' 'urllib3>=2.7.0' 'cryptography>=46.0.7' 'idna>=3.15'
# clean conda and pip caches
RUN rm -rf ~/.cache/pip
COPY loss /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/loss.py
COPY eagle3 /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/eagle3.py
COPY parse /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/data/parse.py
