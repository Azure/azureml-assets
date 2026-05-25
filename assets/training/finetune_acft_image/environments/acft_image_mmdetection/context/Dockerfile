FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# openmim still imports pkg_resources, which is removed from setuptools>=82.
# Keep a temporary compatibility pin for mim installation, then restore setuptools.
RUN pip install --no-cache-dir 'setuptools<82'

# # # Install mmdet
# # Note that MMDet installs pycocotools
# Note: mmdet should be installed via mim to access the model zoo config folder.
RUN mim install mmdet==3.3.0
# Temporary workaround for https://github.com/open-mmlab/mmdetection/issues/11668 (when mmdet updated, remove lines below)
RUN mim install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.2/index.html --no-cache-dir
RUN pip install --no-cache-dir --upgrade setuptools==82.0.0
RUN sed -i 's/2.2.0/2.3.0/' /opt/conda/envs/ptca/lib/python3.10/site-packages/mmdet/__init__.py

# onnx: azureml-acft-accelerator pins onnx<=1.17.0 but that range has known CVEs;
#   onnx-weekly 1.22.0.dev20260504 includes main-branch security fixes (GHSA-3r9x-f23j-gc73,
#   GHSA-538c-55jv-c5g9, GHSA-hqmj-h5c6-369m, etc.) not yet in a stable PyPI release
#   (re-verified 2026-05-23: PyPI latest stable onnx is still 1.21.0).
RUN pip uninstall -y onnx && pip install --no-cache-dir 'onnx-weekly>=1.22.0.dev20260504'

# pip 26.0.1 (GHSA-jp4c-xjxw-mgf9): pip is the Python package installer itself with no upstream parent package,
#   so direct upgrade is the only remediation. pip 26.1+ is on PyPI and conda-forge; the conda `defaults` channel
#   currently tops out at 26.0.1 (checked 2026-05-12), hence `-c conda-forge` is required. Using `conda install`
#   (not `pip install`) so both the dist-info METADATA and the conda-meta JSON are refreshed in one step,
#   ensuring SBOM scanners pick up the new version. `--freeze-installed` keeps the rest of the env intact.
#   Both ptca (py3.10) and base (py3.13) envs ship pip 26.0.1 in the current base image (probed 2026-05-23).
RUN conda install -n ptca -y -c conda-forge --freeze-installed 'pip=26.1.1' \
 && conda install -n base -y -c conda-forge --freeze-installed 'pip=26.1.1'

# Override diffusers 0.24.0 -> 0.38.0 to fix GHSA-98h9-4798-4q5v (CVE-2026-44513).
# Root cause (verified 2026-05 via PyPI metadata against the built image):
#   - diffusers is directly pinned in requirements.txt at ==0.24.0 alongside the rest of the
#     pinned HF stack (transformers==5.5.4, accelerate==0.27.2, optimum==1.23.3,
#     huggingface-hub==1.5.0, datasets==2.15.0) which were chosen for mutual compatibility.
#   - No parent package in the image transitively pins diffusers; it is a direct top-level pin.
#   - Bumping the requirements.txt pin from 0.24.0 to 0.38.0 is a 14-minor-version jump that
#     would pull a new transformers/huggingface_hub/accelerate floor and risks breaking the
#     rest of the pinned HF stack, which is out of scope for a CVE-only patch.
# Using --no-deps to keep the existing HF stack intact; diffusers 0.38.0 runtime imports are
# satisfied by the already-installed transformers/accelerate/huggingface_hub/safetensors.
RUN pip install --no-cache-dir --no-deps 'diffusers==0.38.0'

# Base env (py3.13) overrides for transitive packages still shipped vulnerable by the upstream
# base image (probed on 2026-05-23). The ptca (py3.10) env already
# ships patched versions for these, so only the base env needs the upgrade:
#   - urllib3 2.6.3 -> 2.7.0+ : GHSA-qccp-gfcp-xxvc (CVE-2026-44431),
#     GHSA-mf9v-mfxr-j63j (CVE-2026-44432). Brought in transitively by requests, conda,
#     conda-libmamba-solver, conda_package_streaming; none upper-bound urllib3 below 3,
#     so no parent bump can pull 2.7.0 automatically. Capped to <3 to stay ABI-compatible.
#   - idna 3.11 -> 3.15+      : GHSA-65pc-fj4g-8rjx (CVE-2026-45409). Brought in by
#     anyio, httpx, requests, yarl; none upper-bound idna, no parent bump available.
#   - click 8.2.1 -> 8.3.3+   : GHSA-47fr-3ffg-hgmw (CVE-2026-7246). Brought in by
#     anaconda-cli-base and typer; neither pins click below 8.3.3, so direct upgrade is
#     the only remediation without bumping anaconda tooling versions.
#   - python-dotenv 1.2.1 -> 1.2.2+ : GHSA-mf9w-mj56-hr94. Brought in by
#     anaconda-auth and pydantic-settings; parents do not upper-bound python-dotenv so a
#     parent bump won't force >=1.2.2.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade \
    'urllib3>=2.7.0,<3' \
    'idna>=3.15' \
    'click>=8.3.3' \
    'python-dotenv>=1.2.2'

# Remove vulnerable py-rattler from the base (py3.13) env.
# Root cause (re-verified 2026-05-23 against the built image and PyPI):
#   - The base image ships py-rattler (installed as a conda package, not visible via
#     `pip show`) plus the conda-rattler-solver plugin in /opt/conda. The compiled
#     rattler.abi3.so still bundles vulnerable Rust crates: openssl 0.10.75 (6 CRITICAL/HIGH/MEDIUM CVEs),
#     aws-lc-sys 0.37.1 (3 HIGH + 2 MEDIUM), aws-lc-fips-sys 0.37.1 (HIGH + MEDIUM),
#     bytes 1.11.0 (HIGH), tar 0.4.44 (HIGH + MEDIUM), rustls-webpki 0.103.9 (4 MEDIUM),
#     rand 0.9.2 (MEDIUM) -- ~20 findings total.
#   - py-rattler 0.23.2 is the newest published version on PyPI (checked 2026-05-23),
#     so an upgrade-in-place cannot pick up patched Rust crates yet.
#   - rattler is only the conda dependency solver; ML workloads run in the ptca
#     (py3.10) env and do not import it. Switching conda back to the classic solver
#     keeps `conda` itself working after the rattler packages are removed.
RUN conda config --system --set solver classic \
 && /opt/conda/bin/python3.13 -m pip uninstall -y py-rattler conda-rattler-solver 2>/dev/null || true \
 && rm -rf /opt/conda/lib/python3.13/site-packages/rattler* /opt/conda/lib/python3.13/site-packages/conda_rattler_solver* /opt/conda/lib/python3.13/site-packages/py_rattler* /opt/conda/conda-meta/py-rattler-*.json /opt/conda/conda-meta/conda-rattler-solver-*.json

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
