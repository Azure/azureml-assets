FROM mcr.microsoft.com/azureml/3.5dotnet-ltsc2019
# Ensure base conda env uses older pip which doesn't re-install ruaml package.
RUN pip install --no-cache-dir pip==20.0.2
RUN pip install --no-cache-dir azure-identity==1.7.0
RUN pip install --no-cache-dir azureml-defaults~=1.40.0 azureml-telemetry~=1.40.0
RUN pip install --no-cache-dir azureml-core~=1.40.0.post2
RUN pip install --no-cache-dir azureml-dataprep~=3.0.2
# Install azureml-sidecar dependencies from pypi first to avoid security vulnerabilities.
RUN pip install --no-cache-dir cloudpickle tblib jsonschema
RUN pip install --no-cache-dir --index-url https://azuremlsdktestpypi.azureedge.net/SideCar-Release/D89B478218CC/ azureml-sidecar==2.5.0
# CMDs used by job_prep.py/job_release.py to interact with sidecar daemon.
ENV AZUREML_SIDECAR_PREP_CMD="[\"azureml.sidecar\", \"--task\", \"enter_contexts\"]"
ENV AZUREML_SIDECAR_RELEASE_CMD="[\"azureml.sidecar\", \"--task\", \"exit_contexts\"]"
ENV AZUREML_SIDECAR_IMAGE_NAME="AzureML-Sidecar-Windows"
# Start Sidecar daemon on container is run.
ENTRYPOINT ["python", "-m", "azureml.sidecar", "--task", "start"]
