# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202605.1

RUN apt-get -y update && apt-get -y upgrade
# Install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Flag needed to enable control flow which is in PrP.
ENV AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=True

# vulnerability overrides in base conda env (python3.13)
# setuptools: base image has 82.0.0, need >=82.0.1 for GHSA-58pv-8j8x-9vj2
# python-dotenv: transitive dep of uvicorn[standard] (>=0.13) and pydantic-settings (>=0.21.0); parents use loose floors, cannot force >=1.2.2 via parent (GHSA-mf9w-mj56-hr94)
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'setuptools>=82.0.1' 'python-dotenv>=1.2.2'
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
