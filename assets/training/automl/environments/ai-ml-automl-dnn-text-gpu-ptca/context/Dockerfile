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

# Install/upgrade torch first to avoid conflicts
RUN pip uninstall -y onnxruntime torchaudio || true
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
# Install/upgrade torch in ptca env if it exists (fixes vulnerability)
RUN /opt/conda/envs/ptca/bin/pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 || true

# Install core dependencies first with secure versions
RUN pip install sentencepiece==0.1.97 tokenizers>=0.19.0 huggingface-hub>=0.16.0 --force-reinstall
RUN pip install 'transformers[sentencepiece,torch]==4.53.0' --no-deps --force-reinstall

# Install azureml packages with --no-deps to avoid version conflicts
RUN pip install 'azureml-defaults=={{latest-pypi-version}}' --no-deps --force-reinstall
RUN pip install 'azureml-automl-dnn-nlp=={{latest-pypi-version}}' --no-deps --force-reinstall

# Install compatible onnx and onnxruntime versions
RUN pip install onnx==1.17.0
RUN pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ onnxruntime-training==1.18.0
RUN pip install onnxruntime==1.18.0 --force-reinstall
RUN pip install "numpy==1.22.0" --force-reinstall
RUN /opt/conda/envs/ptca/bin/pip install "numpy==1.22.0" --force-reinstall || true 
RUN TORCH_CUDA_ARCH_LIST="5.2;6.0;7.0;8.0;8.6;9.0" python -m onnxruntime.training.ortmodule.torch_cpp_extensions.install
RUN pip install torch-ort==1.18.0 && TORCH_CUDA_ARCH_LIST="5.2;6.0;7.0;8.0;8.6;9.0" python -m torch_ort.configure

RUN pip install optimum==1.23.3
RUN pip install accelerate==0.33.0
RUN pip install deepspeed~=0.15.1

# Address vulnerabilities - install these after all other packages to override conflicts
RUN pip install pyarrow==14.0.2 --force-reinstall
RUN pip install aiohttp==3.12.14 --force-reinstall
RUN pip install idna==3.7 --force-reinstall
RUN pip install requests==2.32.4 --force-reinstall
# Force install urllib3 last to override any downgrade attempts
RUN pip install urllib3==2.5.0 --force-reinstall --no-deps
RUN opt/conda/bin/pip install --upgrade 'urllib3==2.5.0' --force-reinstall || true

# Patch for Pillow vulnerability
RUN pip install --upgrade 'pillow==11.3.0'
RUN /opt/conda/envs/ptca/bin/pip install --upgrade 'pillow==11.3.0' || true

RUN pip list