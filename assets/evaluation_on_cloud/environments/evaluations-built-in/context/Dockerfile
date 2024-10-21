FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:{{latest-image-tag}}

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy your Python file into the image
COPY evaluate_on_data.py /app/evaluate_on_data.py
COPY save_evaluation.py /app/save_evaluation.py