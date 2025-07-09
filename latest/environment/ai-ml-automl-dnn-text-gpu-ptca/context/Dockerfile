FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu124-py310-torch260:biweekly.202506.3

USER root:root

RUN pip install 'azureml-automl-dnn-nlp==1.60.0'
RUN pip install 'azureml-defaults==1.60.0'

RUN pip uninstall -y onnxruntime
RUN pip install torch==2.7.1 torchvision==0.22.1

RUN pip install onnx==1.17.0
RUN pip uninstall -y onnxruntime-training
RUN pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ onnxruntime-training==1.18.0
RUN TORCH_CUDA_ARCH_LIST="5.2;6.0;7.0;8.0;8.6;9.0" python -m onnxruntime.training.ortmodule.torch_cpp_extensions.install
RUN pip install torch-ort==1.18.0 && TORCH_CUDA_ARCH_LIST="5.2;6.0;7.0;8.0;8.6;9.0" python -m torch_ort.configure

RUN pip uninstall -y onnxruntime

RUN pip install transformers==4.50.0
RUN pip install optimum==1.23.3
RUN pip install accelerate==0.33.0
RUN pip install deepspeed~=0.15.1
RUN pip install numpy==1.22.0

# Address vulnerabilities
RUN pip install pyarrow==14.0.1
RUN pip install aiohttp==3.10.11
RUN pip install idna==3.7
RUN pip install requests==2.32.4
RUN pip install urllib3==2.5.0
RUN opt/conda/bin/pip install --upgrade requests urllib3 || true

RUN pip list