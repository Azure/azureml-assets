FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260715.v1

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Security: starlette is pulled in transitively (via fastapi). Bump to >=1.3.1 to fix
# GHSA-82w8-qh3p-5jfq and GHSA-wqp7-x3pw-xc5r (both HIGH; fixed in 1.3.1). Applied as a
# separate post-install upgrade rather than a requirements.txt pin because forcing the
# floor inside the combined resolution triggers pip "resolution-too-deep". (ICM 831170435)
RUN pip install --no-cache-dir --upgrade 'starlette>=1.3.1'

# Copy your Python file into the image
COPY evaluate_on_data.py /app/evaluate_on_data.py
COPY model_target.py /app/model_target.py
COPY save_evaluation.py /app/save_evaluation.py
ADD online_eval /app/online_eval