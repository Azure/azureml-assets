FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

# Pull current Ubuntu security updates for packages inherited from the base image.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade 'pip>=26.2.0' && \
    python -m pip install --no-cache-dir -r /app/requirements.txt

# Security: starlette is pulled in transitively (via fastapi). Bump to >=1.3.1 to fix
# GHSA-82w8-qh3p-5jfq and GHSA-wqp7-x3pw-xc5r (both HIGH; fixed in 1.3.1). Applied as a
# separate post-install upgrade rather than a requirements.txt pin because forcing the
# floor inside the combined resolution triggers pip "resolution-too-deep". (ICM 831170435)
RUN python -m pip install --no-cache-dir --upgrade 'starlette>=1.3.1'

# Upgrade vulnerable transitive dependencies after the primary dependency graph resolves.
# azureml-mlflow currently caps cryptography below 49; this security floor intentionally
# overrides that stale metadata until the upstream constraint includes cryptography 50.
RUN python -m pip install --no-cache-dir --upgrade \
        'cryptography>=50.0.0' \
        'h2>=4.4.1' \
        'nltk>=3.10.3' \
        'sqlparse>=0.6.0'

# Copy your Python file into the image
COPY evaluate_on_data.py /app/evaluate_on_data.py
COPY model_target.py /app/model_target.py
COPY save_evaluation.py /app/save_evaluation.py
ADD online_eval /app/online_eval