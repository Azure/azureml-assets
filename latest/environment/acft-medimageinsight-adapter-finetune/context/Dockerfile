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
#     pyasn1-modules 0.4.2 (latest 2026-05-23) pins `pyasn1<0.7.0,>=0.6.1` so pip may
#     still resolve <0.6.3 (CVE-2026-30922) — parent does not tighten the floor.
#   Mako: mlflow -> alembic -> Mako; alembic 1.18.4 (latest 2026-05-23) still uses
#     unpinned `Requires-Dist: Mako`.
#   python-dotenv: mlflow -> mlflow-skinny -> python-dotenv<2,>=0.19.0;
#     mlflow-skinny 3.12.0 (latest 2026-05-23) keeps that wide range so pip may resolve
#     <1.2.2 (GHSA-mf9w-mj56-hr94).
#
# base conda env (py3.13):
#   python-dotenv 1.2.1 is brought in by anaconda-auth 0.15.0 (`Requires-Dist: python-dotenv`,
#   no version pin) and pydantic-settings 2.14.1 (`python-dotenv>=0.21.0`); both are shipped
#   pre-installed in the base env from the upstream base image and neither parent ships a
#   tighter pin in its latest release as of 2026-05-23, so we patch python-dotenv directly
#   in the base env via /opt/conda/bin/pip.
RUN pip install --no-cache-dir --upgrade 'pyasn1>=0.6.3' 'Mako>=1.3.11' 'python-dotenv>=1.2.2' \
 && /opt/conda/bin/pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2'

# urllib3 2.6.3 in both base (py3.13) and ptca (py3.10) is vulnerable to
# GHSA-qccp-gfcp-xxvc / CVE-2026-44431 (MEDIUM) and GHSA-mf9v-mfxr-j63j /
# CVE-2026-44432 (HIGH) — fixed in urllib3 2.7.0. urllib3 is a transitive dep
# pulled in by `requests` (used by mlflow, gdown, azureml-* etc.). `requests`
# 2.32.5 (latest 2026-05-23) only requires `urllib3<3,>=1.21.1`, so no parent
# package can be upgraded to pull in 2.7.0 — we must override directly. urllib3
# 2.7.0 is not yet published on PyPI (latest there is 2.6.3) but is available
# on conda-forge, so we install via `conda install` in both envs. We also clear
# any pip-installed 2.6.3 dist-info that conda doesn't track so the SBOM
# scanner does not re-flag the old version.
#
# idna 3.11 in base env (py3.13) is vulnerable to GHSA-65pc-fj4g-8rjx /
# CVE-2026-45409 (CRITICAL) — fixed in 3.15. idna is pulled in transitively by
# `requests` and `httpx`; neither parent pins a floor that forces >=3.15
# (requests 2.32.5 uses `idna<4,>=2.5`). Only present in base env per scan.
#
# click 8.2.1 in base env (py3.13) is vulnerable to GHSA-47fr-3ffg-hgmw /
# CVE-2026-7246 (HIGH) — fixed in 8.3.3. click is bundled in the upstream
# base conda env via several CLI tools (e.g. mlflow, conda CLI plugins); no
# parent package in the base env pins click>=8.3.3 as of 2026-05-23. Only
# present in base env per scan.
RUN conda install -y -n base -c conda-forge 'urllib3=2.7.0' 'idna=3.15' 'click=8.3.3' && \
    conda install -y -n ptca -c conda-forge 'urllib3=2.7.0' && \
    rm -rf /opt/conda/lib/python3.13/site-packages/urllib3-2.6*.dist-info \
           /opt/conda/lib/python3.13/site-packages/idna-3.11.dist-info \
           /opt/conda/lib/python3.13/site-packages/click-8.2*.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/urllib3-2.6*.dist-info && \
    conda clean -ay

# py-rattler bundles a compiled Rust extension (rattler.abi3.so) that statically
# links several Rust crates. The version shipped in the upstream base image
# (mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280, biweekly
# tag) bundles vulnerable crate versions: openssl 0.10.75 (5 CRITICAL CVEs:
# CVE-2026-41676/41677/41678/41681/41898 — fixed in 0.10.78), aws-lc-sys 0.37.1
# (3 HIGH + 2 MEDIUM — fixed in 0.38.0/0.39.0), bytes 1.11.0 (HIGH
# CVE-2026-25541 — fixed in 1.11.1), tar 0.4.44 (HIGH + MEDIUM — fixed in
# 0.4.45), rustls-webpki 0.103.9 (4 MEDIUM — fixed in 0.103.13), and rand
# 0.8.5/0.9.2 (MEDIUM — fixed in 0.8.6/0.9.3). The latest py-rattler on PyPI as
# of 2026-05-22 is 0.23.2 (released 2026-03-20), which still bundles these
# vulnerable crate versions — upgrading does not help. py-rattler is only
# used by conda-rattler-solver as an experimental solver backend for `conda`
# itself; the default solver (libmamba) and classic solver remain available
# and our training/finetune image does not invoke conda at runtime (all
# package installs happen at build time, above). Therefore we remove the
# package and the conda-rattler-solver plugin entirely from the base env to
# eliminate the vulnerable abi3.so. We also clean any conda-meta entries so
# the SBOM scanner does not re-flag the removed package.
RUN /opt/conda/bin/pip uninstall -y py-rattler conda-rattler-solver || true && \
    rm -rf /opt/conda/lib/python3.13/site-packages/rattler \
           /opt/conda/lib/python3.13/site-packages/py_rattler* \
           /opt/conda/lib/python3.13/site-packages/py-rattler* \
           /opt/conda/lib/python3.13/site-packages/conda_rattler_solver* && \
    rm -f /opt/conda/conda-meta/py-rattler-*.json \
          /opt/conda/conda-meta/conda-rattler-solver-*.json

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
