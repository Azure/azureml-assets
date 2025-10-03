FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04@sha256:a80c5cf13d0e829481fb3d28bdd768a36021410ba039987cfa86f3efcb3068d8

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy your Python file into the image
COPY evaluate_on_data.py /app/evaluate_on_data.py
COPY model_target.py /app/model_target.py
COPY save_evaluation.py /app/save_evaluation.py
ADD online_eval /app/online_eval