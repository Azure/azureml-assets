FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

RUN apt-get update && apt-get upgrade -y && \
    apt-get install wget nscd python3-idna git libxml2 curl -y && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install git-lfs -y

RUN wget https://go.dev/dl/go1.24.13.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.24.13.linux-amd64.tar.gz && \
    rm go1.24.13.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:$PATH"

COPY requirements.txt .

# Clean conda cache and force reinstall pip to overwrite metadata
RUN conda clean --all -y
RUN conda install -n base -c conda-forge pip=26.0 "setuptools>=80.0.0" "wheel>=0.46.2" --force-reinstall -y
RUN conda install -n ptca -c conda-forge pip=26.0 "setuptools>=80.0.0" "wheel>=0.46.2" --force-reinstall -y || true

# Force upgrade setuptools and wheel in ptca environment via pip
RUN /opt/conda/envs/ptca/bin/pip install --upgrade --force-reinstall "setuptools>=80.0.0" "wheel>=0.46.2"

RUN rm -rf /opt/conda/lib/python3.13 || true

# This line is added because azureml-automl-dnn-vision depends on python 3.9.
# Added ignore python requires to get this installed.
# RUN pip install --ignore-requires-python azureml-automl-dnn-vision
RUN pip install --no-deps azureml-automl-dnn-vision

RUN pip install -r requirements.txt --no-cache-dir

RUN pip install --upgrade nltk==3.9

# Remove duplicate metadata - keeps newest version only
RUN pip install --upgrade --force-reinstall setuptools wheel && pip cache purge

# Adding this to exclude Python 3.13 packages from the setuptools library fix
RUN rm -rf /opt/conda/lib/python3.13 || true

# List pip packages
RUN pip list

## Delete
RUN rm requirements.txt