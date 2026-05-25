# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202605.2

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

# pip >=26.1.1 patches CVE-2026-6357 / GHSA-jp4c-xjxw-mgf9 (base ships 26.0.1
# in the ptca conda env). pip is the Python package installer itself and has
# NO parent/transitive package that pulls it in (it is bootstrapped by the
# conda distribution), so the only available remediation is to upgrade it
# directly. We must use `conda install` (not `pip install --upgrade pip`)
# because Qualys reads conda-meta JSON records; a pip self-upgrade leaves the
# old conda-meta entry in place and the scanner continues to flag 26.0.1.
# `--no-deps` is used so the conda solver does not refresh the root-env
# `rattler` package (which embeds vulnerable Rust crates in the base image's
# /opt/conda; touching it triggers CRITICAL Rust-side findings unrelated to
# this fix).
RUN conda install -n ptca -c conda-forge -y --no-deps "pip>=26.1.1" && \
    conda clean -afy

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# Downgraded azureml-metrics, to prevent bug related to logger in azureml-metrics. Should be upgraded in next pypi release of acft-multimodal
# Removing it from requirements.txt to avoid conflict with scikit-learn version
RUN pip install azureml-metrics==0.0.33 pyarrow==14.0.1

RUN pip install azureml-acft-common-components==0.0.91
RUN pip install azureml-acft-accelerator==0.0.91

# Final-stage overrides applied AFTER azureml-acft-* installs because those
# packages can pull vulnerable transitive versions back in.
# setuptools: >=82.0.1 patches jaraco.context (GHSA-58pv-8j8x-9vj2); base
#   ships 81.0.0 in ptca and 82.0.0 in base conda env.
# nltk: >=3.9.4 patches GHSA-gfwx-w7gr-fvh7; pulled transitively by
#   rouge-score / sacrebleu which still accept the vulnerable range, so no
#   parent upgrade is available.
# transformers: ensure >=5.0.0 patches GHSA-69w3-r845-3855; requirements.txt
#   pins 5.0.0 but azureml-acft-common-components/accelerator may resolve an
#   older release as a dependency, so the override is reapplied here.
# onnx: >=1.21.0 - onnxruntime-training==1.18.0 (pinned in requirements.txt)
#   has a loose onnx dep that pip otherwise resolves to vulnerable 1.17.0.
# pyOpenSSL: >=26.0.0 patches CVE-2026-27459 - transitive dep of multiple
#   azure-* SDKs; no fixed parent release pins it.
# urllib3: >=2.7.0 patches CVE-2026-44431, CVE-2026-44432 - transitive of
#   requests/azure-* SDKs; no fixed parent release pins it.
RUN pip install --no-cache-dir --upgrade 'setuptools>=82.0.1' 'nltk>=3.9.4' 'transformers>=5.0.0' 'onnx>=1.21.0' 'pyOpenSSL>=26.0.0' 'urllib3>=2.7.0'
# Base conda env (python 3.13 at /opt/conda) overrides. All findings here are
# transitive deps with no single upgradeable parent package in the base image:
#   setuptools >=82.0.1  - patches jaraco.context (GHSA-58pv-8j8x-9vj2)
#   pyOpenSSL  >=26.0.0  - patches CVE-2026-27459; pulled by azure-* SDKs
#   urllib3    >=2.7.0   - patches CVE-2026-44431/44432; pulled by requests
#   python-dotenv >=1.2.2 - GHSA-mf9w-mj56-hr94; mlflow-skinny accepts
#                           >=0.19.0,<2 so no parent upgrade exists.
#   pip        >=26.1.1  - GHSA-jp4c-xjxw-mgf9 / CVE-2026-6357. pip is the
#                          installer itself; no parent. We use pip self-
#                          upgrade (not `conda install`) because the base-env
#                          scanner entry is the dist-info METADATA file
#                          (overwritten by pip), and `conda install` against
#                          the base env would refresh the root `rattler`
#                          package whose embedded Rust crates (openssl 0.10.75,
#                          aws-lc-sys 0.37.1, rustls-webpki 0.103.9, tar
#                          0.4.44, bytes 1.11.0, rand 0.8.5, astral-tokio-tar
#                          0.6.0) introduce HIGH/CRITICAL findings unrelated
#                          to this image's Python surface.
#   idna       >=3.15    - GHSA-65pc-fj4g-8rjx / CVE-2026-45409. Pulled
#                          transitively by requests, cryptography, anyio,
#                          httpx in the base env. requests/cryptography both
#                          accept idna<4 with no upper pin that excludes
#                          3.15, so no parent release pins it.
#   click      >=8.3.3   - GHSA-47fr-3ffg-hgmw / CVE-2026-7246. Pulled
#                          transitively by mlflow-skinny, typer, uvicorn,
#                          jupyter-* installed in the base env. mlflow-skinny
#                          accepts click>=7.0 so no parent release pins a
#                          fixed version.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade \
    'setuptools>=82.0.1' 'pyOpenSSL>=26.0.0' 'urllib3>=2.7.0' \
    'python-dotenv>=1.2.2' 'pip>=26.1.1' 'idna>=3.15' 'click>=8.3.3'
