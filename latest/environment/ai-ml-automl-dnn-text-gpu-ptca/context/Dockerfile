FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202605.2

USER root:root

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove -y

RUN pip install --no-cache-dir \ 
    'azureml-automl-dnn-nlp==1.62.0' \
    'azureml-defaults==1.62.0'

# onnx and onnxruntime-training installation
RUN pip uninstall -y onnxruntime
RUN pip uninstall -y onnxruntime-training
RUN pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ onnxruntime-training==1.18.0

# torch-ort installation
RUN TORCH_CUDA_ARCH_LIST="5.2;6.0;7.0;8.0;8.6;9.0" python -m onnxruntime.training.ortmodule.torch_cpp_extensions.install
RUN pip install torch-ort==1.18.0 && TORCH_CUDA_ARCH_LIST="5.2;6.0;7.0;8.0;8.6;9.0" python -m torch_ort.configure
RUN pip uninstall -y onnxruntime

RUN pip install \ 
    optimum==1.23.3 \
    accelerate==1.12.0 \
    deepspeed~=0.15.1

# Override transformers to fix GHSA-69w3-r845-3855
# Root cause: azureml-automl-dnn-nlp==1.62.0 (latest) pins transformers==4.53.0; cannot upgrade parent
RUN pip install --no-cache-dir --no-deps 'transformers[sentencepiece,torch]==5.5.4'

# Vulnerability patches for ptca environment (python 3.10 at /opt/conda/envs/ptca)
# pip 26.0.1 -> >=26.1.1 (GHSA-jp4c-xjxw-mgf9, CVE-2026-6357): pip is base infra;
#   no parent package brings it, so a direct upgrade is the only fix. The stale
#   conda-meta JSON for pip-26.0.1 is removed so scanners do not re-flag it.
# setuptools 81.0.0 -> >=82.0.1 (GHSA-58pv-8j8x-9vj2 in vendored jaraco.context):
#   setuptools is a build dependency with no parent that pins it.
# pytest >=9.0.3 (GHSA-6w46-j5rx-g56g): ACPT base ptca env not yet patched.
# Override onnx to fix GHSA-cmw6-hcpp-c6jp, GHSA-538c-55jv-c5g9, GHSA-q56x-g2fj-4rj6,
#   GHSA-p433-9wv8-28xj, GHSA-3r9x-f23j-gc73, GHSA-hqmj-h5c6-369m.
#   Root cause: azureml-automl-runtime==1.62.0 (latest) pins onnx<=1.17.0; cannot upgrade parent.
# bokeh 2.4.3 -> >=3.8.2 (GHSA-793v-589g-574v, CVE-2026-21883) and distributed
#   2023.2.0 -> >=2026.1.0 (GHSA-c336-7962-wfj2, CVE-2026-23528) are pulled in by
#   azureml-automl-runtime==1.62.0 which pins bokeh==2.4.3 and distributed==2023.2.0;
#   no newer azureml-automl-runtime release exists, so direct override is required.
RUN /opt/conda/envs/ptca/bin/pip install --upgrade \
    'pip>=26.1.1' \
    'setuptools>=82.0.1' \
    'pytest>=9.0.3' \
    'onnx>=1.21.0' \
    'bokeh>=3.8.2' \
    'distributed>=2026.1.0' && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0.1-*.json

# Vulnerability patches for conda base env (python 3.13 at /opt/conda)
# pip 26.0.1 -> >=26.1.1 (GHSA-jp4c-xjxw-mgf9, CVE-2026-6357): pip is base infra;
#   no parent package brings it, so a direct upgrade is the only fix.
# urllib3 2.6.3 -> >=2.7.0 (GHSA-qccp-gfcp-xxvc / CVE-2026-44431, GHSA-mf9v-mfxr-j63j
#   / CVE-2026-44432): pulled in transitively by `requests` and many conda CLI deps
#   (anaconda-cloud-auth, conda, pip itself). `requests` has no upper bound on urllib3
#   (Requires-Dist: urllib3<3,>=1.21.1), and none of the conda parents pin urllib3
#   tightly, so no parent upgrade can force >=2.7.0 -- direct override required.
# idna >=3.15 (GHSA-65pc-fj4g-8rjx): base env parents (`requests`, `anyio`,
#   `httpx`, `yarl`) declare loose idna requirements, so direct override is required.
# click >=8.3.3 (GHSA-47fr-3ffg-hgmw): base env CLI parents use loose click floors,
#   so direct override is required.
# setuptools 82.0.0 -> >=82.0.1 (GHSA-58pv-8j8x-9vj2 in vendored jaraco.context):
#   no parent package pins setuptools.
# python-dotenv 1.2.1 -> >=1.2.2 (GHSA-mf9w-mj56-hr94): brought in transitively by
#   anaconda-auth==0.14.2 (Requires-Dist: python-dotenv with no version pin) and
#   pydantic-settings==2.12.0 (python-dotenv>=0.21.0, via anaconda-cli-base ->
#   anaconda-auth). Latest releases on PyPI as of 2026-05-19 (anaconda-auth==0.14.4,
#   pydantic-settings==2.14.0) still use the same loose floors, so a parent upgrade
#   cannot force >=1.2.2 -- direct override required.
RUN /opt/conda/bin/pip install --upgrade \
    'pip>=26.1.1' \
    'urllib3>=2.7.0' \
    'idna>=3.15' \
    'click>=8.3.3' \
    'setuptools>=82.0.1' \
    'python-dotenv>=1.2.2'
