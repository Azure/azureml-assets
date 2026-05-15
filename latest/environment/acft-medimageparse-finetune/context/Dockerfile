FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202605.1

USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# pip 26.0.1 in ptca env (GHSA-jp4c-xjxw-mgf9, fixed in pip>=26.1) — verified
# present via probe of base biweekly.202605.1 (2026-05-14). pip is a build/install
# tool with no parent Python package, so an upstream parent upgrade is not
# possible. The conda `defaults` channel only ships up to pip 26.0.1, so we use
# `-c conda-forge` (which has 26.1.1) to keep conda-meta and the /opt/conda/pkgs
# cache in sync; we also delete stray pip-26.0*.dist-info / conda-meta entries
# from prior pip-self-upgrades that conda does not track, otherwise the SBOM
# scanner re-flags them. Done before COPY requirements.txt so requirements and
# detectron2 use the patched pip.
RUN conda install -y -n ptca -c conda-forge pip==26.1.1 && \
    rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/pip-26.0*.dist-info && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json && \
    conda clean -ay

# Base conda env (python 3.13) ships two vulnerable packages independent of the
# ptca env; both confirmed via base-image probe of biweekly.202605.1 (2026-05-14):
#   - pip 26.0.1 at /opt/conda/lib/python3.13/site-packages/pip-26.0.1.dist-info
#     (GHSA-jp4c-xjxw-mgf9, fixed in pip>=26.1) — pip is a tool, no parent
#     upgrade possible.
#   - python-dotenv 1.2.1 at
#     /opt/conda/lib/python3.13/site-packages/python_dotenv-1.2.1.dist-info
#     (GHSA-mf9w-mj56-hr94, fixed in 1.2.2). Parent dependents in this env are
#     `anaconda-auth` (unpinned `python-dotenv`) and `pydantic-settings`
#     (`python-dotenv>=0.21.0`) — both are loose floors, so even latest parents
#     resolve back to 1.2.1. Direct override required until either a) the base
#     ACPT image refreshes to ship python-dotenv>=1.2.2, or b) one of the
#     parents tightens its floor to >=1.2.2.
# python-dotenv in the base env is pip-managed (not conda-managed), so we use
# `python3.13 -m pip install --upgrade` and then remove the stale dist-info
# directories left by the in-place pip self-upgrade so the SBOM scanner does
# not re-flag them.
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade \
        'pip==26.1.1' 'python-dotenv>=1.2.2' && \
    rm -rf /opt/conda/lib/python3.13/site-packages/pip-26.0*.dist-info && \
    rm -rf /opt/conda/lib/python3.13/site-packages/python_dotenv-1.2.1.dist-info

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git@a1ce2f9