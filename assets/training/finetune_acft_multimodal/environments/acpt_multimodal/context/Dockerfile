# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
# linux-headers-* and linux-libc-dev ship multiple HIGH CVEs (CVE-2024-35870,
# CVE-2024-53179, CVE-2024-56692, CVE-2025-37899, CVE-2025-38118) with no
# fixed version available upstream. These kernel build-time packages are
# unnecessary at container runtime (the host provides the kernel), so the
# only available remediation is to purge them.
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get -y purge 'linux-headers-*' 'linux-libc-dev' && \
    apt-get -y autoremove && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# Downgraded azureml-metrics, to prevent bug related to logger in azureml-metrics. Should be upgraded in next pypi release of acft-multimodal
# Removing it from requirements.txt to avoid conflict with scikit-learn version
RUN pip install azureml-metrics==0.0.33 pyarrow==14.0.1

RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install azureml-acft-accelerator=={{latest-pypi-version}}

# setuptools: >=82.0.1 patches jaraco.context (GHSA-58pv-8j8x-9vj2); base ships 82.0.0
# nltk: >=3.9.4 required (GHSA-gfwx-w7gr-fvh7)
# transformers: ensure >=5.0.0 after azureml-* installs (GHSA-69w3-r845-3855)
# onnx: onnxruntime-training has loose onnx dep; pip resolves to 1.17.0 which has multiple CVEs
# pip: >=26.1.1 patches CVE-2026-6357 / GHSA-jp4c-xjxw-mgf9 (base ships 26.0.1).
# pip is the Python package installer itself — it has NO parent/transitive package
# that pulls it in (it is bootstrapped by the conda/python distribution), so the
# only available remediation is to upgrade pip directly in each conda environment.
# pyOpenSSL: >=26.0.0 patches CVE-2026-27459 (transitive dep, force-bump needed).
# urllib3: >=2.7.0 patches CVE-2026-44431, CVE-2026-44432 (transitive dep).
RUN pip install --no-cache-dir --upgrade 'pip>=26.1.1' 'setuptools>=82.0.1' 'nltk>=3.9.4' 'transformers>=5.0.0' 'onnx>=1.21.0' 'pyOpenSSL>=26.0.0' 'urllib3>=2.7.0'
# base conda env: pip, setuptools and python-dotenv need upgrades above what base image ships
# python-dotenv: transitive dep of mlflow-skinny (accepts >=0.19.0,<2); base ships 1.2.1 (GHSA-mf9w-mj56-hr94)
# pip: see note above — standalone bootstrap tool with no parent package; must be upgraded directly
# pyOpenSSL/urllib3: same CVEs as main env (CVE-2026-27459, CVE-2026-44431, CVE-2026-44432)
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'pip>=26.1.1' 'setuptools>=82.0.1' 'python-dotenv>=1.2.2' 'pyOpenSSL>=26.0.0' 'urllib3>=2.7.0'
