# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get -y upgrade \
 && apt-get -y install --only-upgrade \
      liblzma5 \
      xz-utils \
      libgcrypt20 \
      dotnet-host-8.0 \
      dotnet-hostfxr-8.0 \
      dotnet-runtime-8.0 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# transformers: fix GHSA-69w3-r845-3855 (CVE-2026-1839, arbitrary code execution in Trainer).
# --no-deps keeps the pinned HF stack intact (transformers is a direct requirements.txt pin).
RUN pip install --no-cache-dir --no-deps 'transformers==5.5.4'

# vulnerability fixes - this will be removed once we update to MMTracking's latest version
RUN pip install scipy==1.10.1
RUN pip install gunicorn==23.0.0
RUN pip install onnx==1.21.0

# It is recommended to upgrade to numpy version 1.23.5 to address the numpy.core.multiarray issue
RUN pip install numpy==1.23.5

# pin required version of yapf to avoid breaking changes in newer versions
# https://github.com/open-mmlab/mmdetection/issues/10962
RUN pip install yapf==0.40.1

# python-dotenv: base env (py3.13) ships 1.2.1; fix GHSA-mf9w-mj56-hr94 (symlink overwrite).
# Transitive dep with no parent pinning a safe floor; ptca env already gets 1.2.2 via requirements.txt.
# Bound to the 1.x line for reproducible rebuilds.
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2,<2'

# pip 26.0.1 -> 26.1.1 in both envs to fix GHSA-jp4c-xjxw-mgf9 (PEP 770 SBOM tag injection).
# Installed from conda-forge (not pip self-upgrade) so conda-meta is rewritten cleanly and the
# scanner stops flagging the stale 26.0.1 record; defaults stays primary to avoid channel churn.
RUN /opt/conda/bin/conda install -y -n base --override-channels -c defaults -c conda-forge 'conda-forge::pip=26.1.1' \
 && /opt/conda/bin/conda install -y -n ptca --override-channels -c defaults -c conda-forge 'conda-forge::pip=26.1.1' \
 && /opt/conda/bin/conda clean -afy

# py-rattler: pinned transitive dep of conda-rattler-solver brought in by base ACPT image; fixes GHSA-q53q-5r4j-5729.
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'py-rattler>=0.24.0,<0.25'

# cryptography: pinned transitive dep of base ACPT image and AzureML SDK dependencies; fixes GHSA-537c-gmf6-5ccf.
RUN pip install --no-cache-dir --upgrade 'cryptography>=48.0.1,<49' \
 && /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'cryptography>=48.0.1,<49'

# diffusers 0.24.0 -> 0.38.0 to fix GHSA-98h9-4798-4q5v (CVE-2026-44513).
# --no-deps keeps the pinned HF stack intact (diffusers is a direct requirements.txt pin).
RUN pip install --no-cache-dir --no-deps 'diffusers==0.38.0'

# urllib3 2.6.3 -> 2.7.0 in both envs to fix GHSA-qccp-gfcp-xxvc (CVE-2026-44431)
# and GHSA-mf9v-mfxr-j63j (CVE-2026-44432). Transitive dep; bound to 2.x for ABI compatibility.
RUN pip install --no-cache-dir --upgrade 'urllib3>=2.7.0,<3' \
 && /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'urllib3>=2.7.0,<3'

# aiohttp>=3.14.0 (GHSA-jg22-mg44-37j8, GHSA-hg6j-4rv6-33pg), click>=8.3.3 (GHSA-47fr-3ffg-hgmw),
# idna>=3.15 (GHSA-65pc-fj4g-8rjx): transitive deps in both envs, no parent floors the safe versions.
RUN pip install --no-cache-dir --upgrade 'aiohttp>=3.14.0' 'click>=8.3.3' 'idna>=3.15' \
 && /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'aiohttp>=3.14.0' \
 && /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'click>=8.3.3' \
 && /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'idna>=3.15'

# PyJWT>=2.13.0 in base env (py3.13): fix GHSA-jq35-7prp-9v3f, GHSA-993g-76c3-p5m4 (not present in ptca env).
# pyarrow>=23.0.1 in ptca env (py3.10): fix GHSA-rgxp-2hwp-jwgg (API-compatible with 20.x).
# Both are transitive deps with no parent pinning a safe floor.
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir 'PyJWT>=2.13.0' \
 && pip install --no-cache-dir 'pyarrow>=23.0.1'

# pydantic-settings: pinned transitive dep in base ACPT image; fixes GHSA-4xgf-cpjx-pc3j.
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'pydantic-settings==2.14.2'
