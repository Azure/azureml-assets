# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get -y upgrade

RUN apt-get -y install unzip

# pip 26.0.1 in both the base (py3.13) and ptca (py3.10) conda envs is
# vulnerable to GHSA-jp4c-xjxw-mgf9 / CVE-2026-6357 (fixed in pip>=26.1).
# pip is installed by conda from the upstream base image; there is no parent
# Python package that brings it in, so an upstream parent upgrade is not
# possible. The base ACPT image (biweekly.202605.2 as of 2026-05-19) still
# ships pip 26.0.1 in both envs, so we override here. `conda install` is used
# so conda-meta JSON and /opt/conda/pkgs cache are updated, and stale
# pip-26.0*.dist-info / conda-meta entries from prior pip self-upgrades are
# removed (conda does not track those, and the SBOM scanner re-flags them).
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
#   pyasn1 (>=0.6.3, CVE-2026-30922): mlflow -> databricks-sdk -> google-auth
#     -> pyasn1-modules -> pyasn1; pyasn1-modules pins pyasn1 with no version
#     floor so pip may resolve <0.6.3.
#   Mako (>=1.3.11): mlflow -> alembic -> Mako; alembic 1.18.4 uses unpinned
#     `Requires-Dist: Mako`.
#   python-dotenv (>=1.2.2, GHSA-mf9w-mj56-hr94): mlflow -> mlflow-skinny ->
#     python-dotenv<2,>=0.19.0; mlflow 3.11.1 (latest as of 2026-05-19) keeps
#     that wide range.
#   (urllib3 is already 2.7.0 in the ptca env of biweekly.202605.2, so no
#    override is needed there.)
#
# base conda env (py3.13):
#   python-dotenv 1.2.1 is brought in by anaconda-auth 0.14.4
#     (`Requires-Dist: python-dotenv`, no version pin) and pydantic-settings
#     2.12.0 (`python-dotenv>=0.21.0`); both are pre-installed in the base env
#     from the upstream base image and neither parent ships a tighter pin in
#     its latest release as of 2026-05-19, so we patch python-dotenv directly.
#   urllib3 2.6.3 (GHSA-qccp-gfcp-xxvc / CVE-2026-44431,
#     GHSA-mf9v-mfxr-j63j / CVE-2026-44432; fixed in 2.7.0) is brought in by
#     `requests` (`urllib3<3,>=1.26`). requests 2.32.5 (latest as of
#     2026-05-19) keeps that wide range so no parent upgrade can pull in the
#     patched urllib3 — override directly via /opt/conda/bin/pip.
RUN pip install --no-cache-dir --upgrade 'pyasn1>=0.6.3' 'Mako>=1.3.11' 'python-dotenv>=1.2.2' \
 && /opt/conda/bin/pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2' 'urllib3>=2.7.0'

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/