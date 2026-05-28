FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

# Fix S360 vulnerabilities: update OS packages (openssh, curl, kmod, sed, nghttp2, dpkg)
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Upgrade pip and GitPython to fix known CVEs (after requirements to avoid downgrade)
RUN pip install --upgrade "pip==26.1" "GitPython==3.1.50"

# Copy your Python file into the image
COPY evaluate_on_data.py /app/evaluate_on_data.py
COPY model_target.py /app/model_target.py
COPY save_evaluation.py /app/save_evaluation.py
ADD online_eval /app/online_eval