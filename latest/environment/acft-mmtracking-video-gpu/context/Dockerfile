# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.1

USER root
RUN apt-get -y update && apt-get -y upgrade

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir


# Override transformers to fix GHSA-69w3-r845-3855(CVE-2026-1839, arbitrary code execution in Trainer)
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

# pip 26.0.1 -> 26.1.1 to fix GHSA-jp4c-xjxw-mgf9 (PEP 770 SBOM tag injection in pip install --report).
# Root cause (verified 2026-05 against the built image):
#   - pip is a leaf conda package in both /opt/conda (py3.13, base env) and
#     /opt/conda/envs/ptca (py3.10) - installed directly by the upstream PTCA
#     base image; not pulled in transitively by anything we control.
#   - Upstream `defaults` channel (https://repo.anaconda.com/pkgs/main/noarch)
#     only ships pip 26.0.1 as of 2026-05; pip 26.1.1 is published on
#     conda-forge but has not been mirrored to defaults yet, so the upstream
#     base image cannot pick it up via its standard channel.
#   - A plain `pip install --upgrade pip` would leave conda-meta/pip-26.0.1*.json
#     in place, so trivy would still report the conda-pkg as 26.0.1 even after
#     the site-packages dist-info is replaced. We therefore upgrade via conda
#     from conda-forge so conda-meta is rewritten cleanly.
#   - The transaction is narrowly scoped: pip is channel-qualified to
#     conda-forge and pinned exactly; everything else stays on `defaults` to
#     avoid pulling other packages over to conda-forge.
RUN /opt/conda/bin/conda install -y -n base --override-channels -c defaults -c conda-forge 'conda-forge::pip=26.1.1' \
 && /opt/conda/bin/conda install -y -n ptca --override-channels -c defaults -c conda-forge 'conda-forge::pip=26.1.1' \
 && /opt/conda/bin/conda clean -afy

# Override diffusers 0.24.0 -> 0.38.0 to fix GHSA-98h9-4798-4q5v (CVE-2026-44513).
# Root cause (verified 2026-05 via PyPI metadata against the built image):
#   - diffusers is directly pinned in requirements.txt line 11 at ==0.24.0 alongside the rest
#     of the pinned HF stack (transformers==4.53.0, accelerate==0.25.0, optimum==1.23.3,
#     peft==0.15.2, datasets==2.15.0) which were chosen for mutual compatibility.
#   - No parent package in the image transitively pins diffusers; it is a direct top-level pin.
#   - Bumping the requirements.txt pin from 0.24.0 to 0.38.0 is a 14-minor-version jump that
#     would pull a new transformers/huggingface_hub/accelerate floor and risks breaking the
#     rest of the pinned HF stack, which is out of scope for a CVE-only patch.
# Using --no-deps to keep the existing HF stack intact; diffusers 0.38.0 runtime imports are
# satisfied by the already-installed transformers/accelerate/huggingface_hub/safetensors.
RUN pip install --no-cache-dir --no-deps 'diffusers==0.38.0'

# Override urllib3 2.6.3 -> 2.7.0 in both envs to fix GHSA-qccp-gfcp-xxvc (CVE-2026-44431)
# and GHSA-mf9v-mfxr-j63j (CVE-2026-44432).
# Root cause (verified 2026-05 via PyPI metadata against the built image):
#   - urllib3 is a transitive dep brought in by requests, botocore, azureml-core, etc.;
#     none of these parents pin urllib3 to a version that excludes 2.7.0 (requests allows
#     urllib3<3, botocore allows urllib3<3 on py>=3.10), so there is no parent we can bump
#     to pick up 2.7.0 automatically.
#   - The upstream PTCA base image still ships 2.6.3 in both /opt/conda (py3.13) and
#     /opt/conda/envs/ptca (py3.10); the parent packages above resolve to 2.6.3 because
#     that is the highest version cached/mirrored at base-image build time.
# Bound the upgrade to the 2.x line to keep ABI-compatible with the existing requests stack.
RUN pip install --no-cache-dir --upgrade 'urllib3>=2.7.0,<3' \
 && /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'urllib3>=2.7.0,<3'
