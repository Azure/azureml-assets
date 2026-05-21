FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# pip override for the ptca env (python 3.10):
# Base ships pip 26.0.1 (GHSA-jp4c-xjxw-mgf9, fixed in pip>=26.1) — confirmed
# via base probe of biweekly.202605.2 (2026-05-19). pip is a self-bootstrapping
# install tool with no upstream parent package, so a parent-only upgrade is not
# possible. The conda `defaults` channel only ships up to pip 26.0.1, so we use
# `-c conda-forge` (26.1.1) to keep conda-meta and /opt/conda/pkgs in sync.
# We also delete stray pip-26.0*.dist-info / conda-meta entries that prior
# pip-self-upgrades leave behind, otherwise the SBOM scanner re-flags them.
RUN conda install -y -n ptca -c conda-forge pip==26.1.1 && \
    rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/pip-26.0*.dist-info && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json && \
    conda clean -ay

# Base conda env (python 3.13) overrides — verified against base probe of
# biweekly.202605.2 (2026-05-19):
#   - pip 26.0.1 (GHSA-jp4c-xjxw-mgf9, fixed in pip>=26.1). pip is a tool, no
#     parent upgrade possible.
#   - python-dotenv 1.2.1 (GHSA-mf9w-mj56-hr94, fixed in 1.2.2). Parent
#     dependents in this env are `anaconda-auth` (unpinned) and
#     `pydantic-settings` (`python-dotenv>=0.21.0`) — both loose floors, so
#     latest parents resolve back to 1.2.1. Direct override required until the
#     base image ships python-dotenv>=1.2.2.
#   - urllib3 2.6.3 (GHSA-qccp-gfcp-xxvc, GHSA-mf9v-mfxr-j63j; fixed in 2.7.0).
#     The base env has no narrow parent pin; urllib3 here is shipped by the
#     base conda env itself. The latest `requests` (2.32.5) and `botocore`
#     allow urllib3>=2 with no upper, so a parent-only upgrade still resolves
#     to whatever the base env has (2.6.3). Direct override until the base
#     image refreshes to urllib3>=2.7.0. (urllib3 in the ptca env is already
#     2.7.0 in the base, so no ptca override is needed.)
# All three packages are pip-managed in the base env (not conda-managed).
# Pip's in-place self-upgrade leaves an orphan pip-26.0*.dist-info that the
# SBOM scanner re-flags, so we delete it after the upgrade. urllib3 and
# python-dotenv are regular packages and pip cleans their old dist-info
# normally — no extra rm needed for those.
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade \
        'pip==26.1.1' 'python-dotenv>=1.2.2' 'urllib3>=2.7.0' && \
    rm -rf /opt/conda/lib/python3.13/site-packages/pip-26.0*.dist-info

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git@a1ce2f9
