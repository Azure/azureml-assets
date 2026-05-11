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
# python-dotenv: transitive dep of pydantic-settings (requires python-dotenv>=0.21.0) and anaconda-auth (requires python-dotenv with no version floor);
#   even latest pydantic-settings 2.14.0 and anaconda-auth 0.14.4 use the same loose floors, so >=1.2.2 cannot be forced via parents (GHSA-mf9w-mj56-hr94)
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'setuptools>=82.0.1' 'python-dotenv>=1.2.2'
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
