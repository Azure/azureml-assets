FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu113-py38-torch1110:monthly.202206

COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge
