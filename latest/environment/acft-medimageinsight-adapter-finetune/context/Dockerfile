# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202605.2

USER root
RUN apt-get -y update && apt-get -y upgrade

RUN apt-get -y install unzip

# pip 26.0.1 in both the base (py3.13) and ptca (py3.10) conda envs is
# vulnerable to GHSA-jp4c-xjxw-mgf9 / CVE-2026-6357 (fixed in pip>=26.1).
# pip is a build/install tool installed by conda from the upstream base image
# (mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280, biweekly
# tag) — there is no parent Python package that brings it in, so an upstream
# parent upgrade is not possible. The base ACPT image has not yet refreshed to
# pip 26.1+ as of 2026-05-12, so we override here. We use `conda install`
# (rather than `pip install --upgrade`) so that conda-meta JSON and
# /opt/conda/pkgs cache are also updated, and we additionally remove stray
# pip-26.0*.dist-info / conda-meta entries from prior pip self-upgrades that
# conda does not track — otherwise the SBOM scanner re-flags them. Done before
# the requirements install so requirements are installed with the patched pip.
RUN conda install -y -n base -c conda-forge pip==26.1.1 && \
    conda install -y -n ptca -c conda-forge pip==26.1.1 && \
    rm -rf /opt/conda/lib/python3.13/site-packages/pip-26.0*.dist-info && \
    rm -f /opt/conda/conda-meta/pip-26.0*.json && \
    rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/pip-26.0*.dist-info && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json && \
    conda clean -ay

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Transitive dep overrides where pip may not resolve to the patched version.
# The image has two conda envs: base (py3.13) and ptca (py3.10); the default
# `pip` resolves to the ptca env, so base-env overrides need /opt/conda/bin/pip.
#
# ptca env (py3.10):
#   pyasn1: mlflow -> databricks-sdk -> google-auth -> pyasn1-modules -> pyasn1;
#     pyasn1-modules pins pyasn1 with no version floor so pip may resolve <0.6.3 (CVE-2026-30922)
#   Mako: mlflow -> alembic -> Mako; alembic 1.18.4 uses unpinned `Requires-Dist: Mako`
#   python-dotenv: mlflow -> mlflow-skinny -> python-dotenv<2,>=0.19.0;
#     mlflow 3.11.1 (latest as of 2026-05-08) keeps that wide range so pip may resolve <1.2.2 (GHSA-mf9w-mj56-hr94)
#
# base conda env (py3.13):
#   python-dotenv 1.2.1 is brought in by anaconda-auth 0.13.1 (`Requires-Dist: python-dotenv`,
#   no version pin) and pydantic-settings 2.12.0 (`python-dotenv>=0.21.0`); both are shipped
#   pre-installed in the base env from the upstream base image and neither parent ships a
#   tighter pin in its latest release as of 2026-05-08, so we patch python-dotenv directly
#   in the base env via /opt/conda/bin/pip.
RUN pip install --no-cache-dir --upgrade 'pyasn1>=0.6.3' 'Mako>=1.3.11' 'python-dotenv>=1.2.2' \
 && /opt/conda/bin/pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2'

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/