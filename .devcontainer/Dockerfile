FROM mcr.microsoft.com/azure-cli
RUN echo "pip install "azure-ml-component[notebooks]" --upgrade" | bash
RUN echo "az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/componentsdk/azure_cli_ml-0.9.8-py3-none-any.whl --pip-extra-index-urls https://azuremlsdktestpypi.azureedge.net/componentsdk/0.9.8 --yes --verbose" | bash
RUN echo "export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true" >> /root/.bashrc
