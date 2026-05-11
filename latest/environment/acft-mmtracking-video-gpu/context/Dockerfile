# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202605.1

USER root
RUN apt-get -y update && apt-get -y upgrade

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir


# [Temp Fix] To address PyTorch Distributed RPC Framework Remote Code Execution (RCE) Vulnerability - Working internally to get a compatbile image, will re-enable
# RUN mim install mmtrack==0.14.0
# RUN mim install mmcv-full==1.7.1
# # # Note that MMDet installs pycocotools
# # Note: mmdet should be installed via mim to access the model zoo config folder.
# RUN mim install mmdet==2.28.2

# Override transformers to fix GHSA-69w3-r845-3855 (CVE-2026-1839, arbitrary code execution in Trainer)
# Root cause (verified 2026-05 via importlib.metadata against the built image):
#   - requirements.txt line 8 directly pins `transformers==4.53.0` for compatibility with the
#     pinned HF stack (accelerate==0.25.0, optimum==1.23.3, diffusers==0.24.0, peft==0.15.2).
#   - azureml-acft-image-components==0.0.89 (current latest) only constrains transformers via
#     optional extras (hf/all -> ==4.33.0; mip/mip3d -> >=4.40); none of those extras are
#     activated in this image, so the parent does NOT block an upgrade.
#   - Bumping the explicit requirements.txt pin to 5.x is a major-version jump that risks
#     breaking the rest of the pinned HF stack and is out of scope for a CVE-only patch.
# Using --no-deps to avoid pulling incompatible transitive deps for the older HF stack.
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

# python-dotenv 1.2.1 in system Python (3.13) needs upgrade to 1.2.2 (GHSA-mf9w-mj56-hr94, symlink overwrite).
# Root cause (verified 2026-05 via importlib.metadata against /opt/conda py3.13):
#   - python-dotenv is installed in the base /opt/conda env (py3.13) by the upstream
#     base image, brought in transitively by anaconda-auth (no version constraint) and
#     pydantic-settings (>=0.21.0); both parents accept any 1.x including 1.2.2.
#   - No parent in /opt/conda blocks the upgrade, but neither parent has been re-released
#     to require >=1.2.2 yet, so we must override here until the base image catches up.
#   - The ptca conda env (py3.10) already gets 1.2.2 transitively via mlflow-skinny when
#     requirements.txt is installed above, so this RUN only needs to fix the py3.13 env.
# Bound the upgrade to the 1.x line to keep rebuilds reproducible without locking out future patches.
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2,<2'
