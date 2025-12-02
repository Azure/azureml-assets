# =========================================
# Builder: compile ORAS with patched Go
# =========================================
FROM ubuntu:24.04 AS oras-builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*
ARG GO_VERSION=1.25.3
RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o /tmp/go.tar.gz \
 && tar -C /usr/local -xzf /tmp/go.tar.gz \
 && rm /tmp/go.tar.gz
ENV PATH=/usr/local/go/bin:$PATH
ARG ORAS_VERSION=1.3.0
# RUN GOBIN=/out go install oras.land/oras/cmd/oras@v${ORAS_VERSION} \
#  && test -x /out/oras
# Build ORAS v1.3.0 with patched golang.org/x/crypto to fix:
# CVE-2025-58181 (GHSA-j5w8-q4qc-rx2x) and CVE-2025-47914 (GHSA-f6x5-jh6r-wrfv)
RUN git clone --depth 1 --branch v${ORAS_VERSION} https://github.com/oras-project/oras.git /tmp/oras \
 && cd /tmp/oras \
 && go get golang.org/x/crypto@latest \
 && go mod tidy \
 && GOBIN=/out go install ./cmd/oras \
 && test -x /out/oras \
 && rm -rf /tmp/oras
 
# =========================================
# Runtime: AzureML base + Docker + Trivy + conda
# =========================================
FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20251130.v1
ENV DEBIAN_FRONTEND=noninteractive

# Docker APT repo + minimal install (no recommends) + cleanup
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
 && chmod a+r /etc/apt/keyrings/docker.asc \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
   $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
   | tee /etc/apt/sources.list.d/docker.list > /dev/null \
 && apt-get update \
 && apt-get install -y --no-install-recommends docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
 && rm -rf /var/lib/apt/lists/*

# Trivy install (same small-footprint pattern)
ARG TRIVY_VERSION=0.67.2
RUN curl -fsSL -o /tmp/trivy.deb "https://github.com/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}/trivy_${TRIVY_VERSION}_Linux-64bit.deb" \
 && apt-get update \
 && apt-get install -y --no-install-recommends /tmp/trivy.deb \
 && rm -f /tmp/trivy.deb \
 && rm -rf /var/lib/apt/lists/* \
 && trivy image --download-db-only --cache-dir /tmp || true \
 && trivy image --download-java-db-only --cache-dir /tmp || true \
 && rm -rf /tmp/*

# ORAS from builder (compiled with patched Go)
COPY --from=oras-builder /out/oras /usr/local/bin/oras
RUN chmod 0755 /usr/local/bin/oras \
 && /usr/local/bin/oras version || true

# AzureML conda env (minimal) + cleanup
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/image-build
RUN conda create -y -p "$AZUREML_CONDA_ENVIRONMENT_PATH" python=3.11 pip=25.* -c conda-forge \
 && conda clean -afy
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Pip deps (pin + no cache)
RUN pip install --no-cache-dir azure-storage-blob==12.20.0
