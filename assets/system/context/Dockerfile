# Shared Go toolchain version used across builder stages; compiled from source so the
# trivy and oras binaries link a patched Go stdlib. Bump to the latest patched Go.
ARG GO_VERSION=1.26.4
ARG TRIVY_VERSION=0.72.0
ARG ORAS_VERSION=1.3.2

# Mode toggles: "build" (compile from source with patched Go + deps) or
# "install" (download the upstream release binary). Both default to "build"
# because the upstream releases currently ship vulnerable Go modules / stdlib.
# Flip an individual ARG to "install" at build time (e.g.
# `--build-arg TRIVY_MODE=install`) once upstream ships a clean release.
ARG TRIVY_MODE=build
ARG ORAS_MODE=build

# =========================================
# Builder: compile ORAS with patched Go
# =========================================
FROM ubuntu:24.04 AS oras-build
ARG GO_VERSION
ARG ORAS_VERSION
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o /tmp/go.tar.gz \
 && tar -C /usr/local -xzf /tmp/go.tar.gz \
 && rm /tmp/go.tar.gz
ENV PATH=/usr/local/go/bin:$PATH
# Build ORAS; pin patched module versions on top of the release tag for any dep the
# tagged release still carries that the scanner flags.
RUN git clone --depth 1 --branch v${ORAS_VERSION} https://github.com/oras-project/oras.git /tmp/oras \
 && cd /tmp/oras \
 && go get golang.org/x/crypto@latest \
 && go get oras.land/oras-go/v2@v2.6.1 \
 && go mod tidy \
 && GOBIN=/out go install ./cmd/oras \
 && test -x /out/oras \
 && rm -rf /tmp/oras

# =========================================
# Installer: download official ORAS release binary
# =========================================
FROM ubuntu:24.04 AS oras-install
ARG ORAS_VERSION
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL -o /tmp/oras.tar.gz \
      "https://github.com/oras-project/oras/releases/download/v${ORAS_VERSION}/oras_${ORAS_VERSION}_linux_amd64.tar.gz" \
 && mkdir -p /out \
 && tar -C /out -xzf /tmp/oras.tar.gz oras \
 && chmod 0755 /out/oras \
 && test -x /out/oras \
 && rm -f /tmp/oras.tar.gz

# Alias stage chosen by ORAS_MODE (build | install). BuildKit only executes the
# referenced upstream stage; the other is pruned from the build graph.
FROM oras-${ORAS_MODE} AS oras-final

# =========================================
# Builder: compile Trivy with patched Go + deps
# =========================================
# Build Trivy from source so it links a patched Go stdlib (GO_VERSION). Prefer bumping
# TRIVY_VERSION to the latest release (its go.mod carries fixed modules) over hand-pinning
# deps; add a `go get pkg@version` below only for a module the latest release still lags.
FROM ubuntu:24.04 AS trivy-build
ARG GO_VERSION
ARG TRIVY_VERSION
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o /tmp/go.tar.gz \
 && tar -C /usr/local -xzf /tmp/go.tar.gz \
 && rm /tmp/go.tar.gz
ENV PATH=/usr/local/go/bin:$PATH
# Build env replicates trivy's magefile (CGO disabled, jsonv2 experiment, version ldflag).
RUN git clone --depth 1 --branch v${TRIVY_VERSION} https://github.com/aquasecurity/trivy.git /tmp/trivy \
 && cd /tmp/trivy \
 && go get oras.land/oras-go/v2@v2.6.1 \
 && go mod tidy \
 && CGO_ENABLED=0 GOEXPERIMENT=jsonv2 go build \
      -ldflags="-s -w -X=github.com/aquasecurity/trivy/pkg/version/app.ver=${TRIVY_VERSION}" \
      -o /out/trivy ./cmd/trivy \
 && test -x /out/trivy \
 && /out/trivy --version \
 && echo "--- linked Go module versions (verifying CVE-fix pins) ---" \
 && go version -m /out/trivy | grep -E '(stdlib|oras-go|rekor|golang\.org/x/(crypto|net)|in-toto-golang|go-git/v5|go-billy/v5|containerd)' \
 && rm -rf /tmp/trivy

# =========================================
# Installer: extract Trivy from the official .deb release
# =========================================
FROM ubuntu:24.04 AS trivy-install
ARG TRIVY_VERSION
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL -o /tmp/trivy.deb \
      "https://github.com/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}/trivy_${TRIVY_VERSION}_Linux-64bit.deb" \
 && mkdir -p /out /tmp/trivy-extract \
 && dpkg-deb -x /tmp/trivy.deb /tmp/trivy-extract \
 && cp /tmp/trivy-extract/usr/bin/trivy /out/trivy \
 && chmod 0755 /out/trivy \
 && test -x /out/trivy \
 && rm -rf /tmp/trivy.deb /tmp/trivy-extract

# Alias stage chosen by TRIVY_MODE (build | install). BuildKit only executes the
# referenced upstream stage; the other is pruned from the build graph.
FROM trivy-${TRIVY_MODE} AS trivy-final

# =========================================
# Runtime: AzureML base + Docker + Trivy + conda
# =========================================
FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}
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

# Trivy from the mode-selected stage (build | install).
# Symlinked at /usr/bin/trivy too so anything assuming the apt-install location keeps working.
COPY --from=trivy-final /out/trivy /usr/local/bin/trivy
RUN chmod 0755 /usr/local/bin/trivy \
 && ln -sf /usr/local/bin/trivy /usr/bin/trivy \
 && /usr/local/bin/trivy --version \
 && trivy image --download-db-only --cache-dir /tmp || true \
 && trivy image --download-java-db-only --cache-dir /tmp || true \
 && rm -rf /tmp/*

# ORAS from the mode-selected stage (build | install).
COPY --from=oras-final /out/oras /usr/local/bin/oras
RUN chmod 0755 /usr/local/bin/oras \
 && /usr/local/bin/oras version || true

# AzureML conda env (minimal) + cleanup
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/image-build
RUN conda create -y -p "$AZUREML_CONDA_ENVIRONMENT_PATH" python=3.11 pip=26.* -c conda-forge \
 && conda clean -afy
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Pip deps (pin + no cache)
RUN pip install --no-cache-dir azure-storage-blob==12.20.0
