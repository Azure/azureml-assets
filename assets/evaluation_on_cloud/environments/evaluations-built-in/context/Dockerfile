FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy your Python file into the image
COPY evaluate.py /app/evaluate.py
COPY save_evaluation.py /app/save_evaluation.py