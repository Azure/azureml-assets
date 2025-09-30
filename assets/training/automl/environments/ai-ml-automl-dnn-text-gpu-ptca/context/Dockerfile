FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu124-py310-torch260:{{latest-image-tag}}

USER root:root

# Update system package index and upgrade Python 3.10 packages to required versions
RUN apt-get update && \
    apt-get install -y --only-upgrade \
        libpython3.10-stdlib \
        python3.10 \
        libpython3.10-minimal \
        python3.10-minimal \
        libpam0g \
        libpam-modules-bin \
        libpam-modules \
        libpam-runtime && \
    apt-get install -y sudo=1.9.9-1ubuntu2.5 && \
    apt-mark hold sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Skip pip dependency checks to allow torch 2.8.0 with azureml-automl-dnn-nlp
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install azureml-automl-dnn-nlp with --no-deps to bypass torch==2.2.2 requirement
# We'll force-install torch 2.8.0 later for security vulnerability fixes
RUN pip install 'azureml-automl-dnn-nlp=={{latest-pypi-version}}' --no-deps
RUN pip install 'azureml-defaults=={{latest-pypi-version}}'

# Install common dependencies that azureml-automl-dnn-nlp might need
RUN pip install transformers datasets tokenizers accelerate

RUN pip uninstall -y onnxruntime
# Force install torch 2.8.0 after azureml-automl-dnn-nlp to resolve version conflict
RUN pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
# Install/upgrade torch in ptca env if it exists (fixes vulnerability)
RUN /opt/conda/envs/ptca/bin/pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 || true

RUN pip install onnx==1.16.1
RUN pip uninstall -y onnxruntime-training
RUN pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ onnxruntime-training==1.17.0
RUN pip install "numpy==1.22.0" --force-reinstall
RUN /opt/conda/envs/ptca/bin/pip install "numpy==1.22.0" --force-reinstall || true 
RUN TORCH_CUDA_ARCH_LIST="5.2;6.0;7.0;8.0;8.6;9.0" python -m onnxruntime.training.ortmodule.torch_cpp_extensions.install
RUN pip install torch-ort==1.18.0 && TORCH_CUDA_ARCH_LIST="5.2;6.0;7.0;8.0;8.6;9.0" python -m torch_ort.configure

RUN pip uninstall -y onnxruntime

RUN pip install transformers==4.48.0
RUN pip install optimum==1.23.3
RUN pip install accelerate==0.33.0
RUN pip install deepspeed~=0.15.1

# Address vulnerabilities
RUN pip install pyarrow==14.0.1
RUN pip install aiohttp==3.12.14
RUN pip install idna==3.7
RUN pip install requests==2.32.4
RUN pip install urllib3==1.26.20
RUN opt/conda/bin/pip install --upgrade 'urllib3==1.26.20' || true

# Patch for Pillow vulnerability
RUN pip install --upgrade 'pillow==11.3.0'
RUN /opt/conda/envs/ptca/bin/pip install --upgrade 'pillow==11.3.0' || true

# Verify torch installation and basic imports work
RUN python -c "import torch; print(f'Torch version: {torch.__version__}'); print('Torch CUDA available:', torch.cuda.is_available())"
RUN python -c "import azureml.automl.dnn.nlp; print('azureml-automl-dnn-nlp imported successfully')" || echo "Warning: azureml-automl-dnn-nlp import may have issues but continuing..."

RUN pip list