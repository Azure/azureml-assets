FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202605.2

USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ptca env (python 3.10) override:
#   pip 26.0.1 ships in base (GHSA-jp4c-xjxw-mgf9, fixed in pip>=26.1) — verified
#   via base probe of biweekly.202605.2 on 2026-05-23. pip is a self-bootstrapping
#   install tool with no upstream parent package, so a parent-only upgrade is not
#   possible. The conda `defaults` channel only ships up to pip 26.0.1, so we use
#   `-c conda-forge` (26.1.1) to keep conda-meta and /opt/conda/pkgs in sync.
#   Stray pip-26.0*.dist-info / conda-meta entries left behind after the swap are
#   removed so the SBOM scanner does not re-flag them.
RUN conda install -y -n ptca -c conda-forge pip==26.1.1 && \
    rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/pip-26.0*.dist-info && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json && \
    conda clean -ay

# Base conda env (python 3.13) overrides — verified against base probe of
# biweekly.202605.2 on 2026-05-23:
#   - pip 26.0.1 (GHSA-jp4c-xjxw-mgf9, fixed in pip>=26.1). pip is a tool, no
#     parent upgrade possible.
#   - python-dotenv 1.2.1 (GHSA-mf9w-mj56-hr94, fixed in 1.2.2). Parent
#     dependents in the base env are `anaconda-auth` (unpinned) and
#     `pydantic-settings` (`python-dotenv>=0.21.0`) — both loose floors, so a
#     parent-only upgrade still resolves back to 1.2.1. Direct override required
#     until the base image ships python-dotenv>=1.2.2.
#   - urllib3 2.6.3 (GHSA-qccp-gfcp-xxvc, GHSA-mf9v-mfxr-j63j; fixed in 2.7.0).
#     Required-by `requests` (no upper pin) — parent upgrade does not help.
#     Direct override until base image refreshes to urllib3>=2.7.0. (ptca env
#     already ships urllib3 2.7.0 in this base, so no ptca override is needed.)
# All three packages are pip-managed in the base env. Pip's in-place self-upgrade
# leaves an orphan pip-26.0*.dist-info that the SBOM scanner re-flags, so we
# delete it after the upgrade. urllib3 and python-dotenv are regular packages
# and pip cleans their old dist-info normally.
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade \
        'pip==26.1.1' 'python-dotenv>=1.2.2' 'urllib3>=2.7.0' && \
    rm -rf /opt/conda/lib/python3.13/site-packages/pip-26.0*.dist-info

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git@a1ce2f9
