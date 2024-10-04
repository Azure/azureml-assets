FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your Python file into the image
COPY evaluate_on_data.py .
COPY save_evaluation.py .