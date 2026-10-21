# Build the source-compiled tools with Go 1.26.10 (2026-10-19 security release) to block the
# untrusted certificate chains covered by CVE-2026-111943. Stay on the supported
# 1.26 line when applying later security updates.
ARG GO_VERSION=1.26.10
ARG TRIVY_VERSION=0.74.0
ARG ORAS_VERSION=1.3.4
ARG BUILDX_VERSION=0.38.1
ARG COMPOSE_VERSION=5.6.0
ARG CONTAINERD_VERSION=2.4.2
ARG RUNC_VERSION=1.5.2

# Mode toggles: "build" (compile from source with patched Go + deps) or
# "install" (download the upstream release binary). Both default to "build"
# because the upstream releases currently ship vulnerable Go modules / stdlib.
# Flip an individual ARG to "install" at build time (e.g.
# `--build-arg TRIVY_MODE=install`) once upstream ships a clean release.
ARG TRIVY_MODE=build
ARG ORAS_MODE=build

# =========================================
# Shared toolchain for source-built tools
# =========================================
FROM ubuntu:24.04 AS go-build
ARG GO_VERSION
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o /tmp/go.tar.gz \
 && tar -C /usr/local -xzf /tmp/go.tar.gz \
 && rm /tmp/go.tar.gz
ENV PATH=/usr/local/go/bin:$PATH GOTOOLCHAIN=local

FROM go-build AS oras-build
ARG ORAS_VERSION
# Build ORAS; pin patched module versions on top of the release tag for any dep the
# tagged release still carries that the scanner flags.
RUN git clone --depth 1 --branch v${ORAS_VERSION} https://github.com/oras-project/oras.git /tmp/oras \
 && cd /tmp/oras \
 && go get golang.org/x/crypto@latest \
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
# TRIVY_VERSION to a release carrying fixed modules when it supports the selected Go
# line. Otherwise apply targeted module overrides without changing the tool's behavior.
FROM go-build AS trivy-build
ARG TRIVY_VERSION
# Build env replicates trivy's magefile (CGO disabled, jsonv2 experiment, version ldflag).
# Trivy v0.74.0 still needs these module fixes:
# grpc 1.83.2: CVE-2026-84304, CVE-2026-84303, CVE-2026-84445.
# x/crypto 0.57.0: retains the CVE-2026-78662/CVE-2026-56855 fixes and is
# required by x/net 0.60.0.
# x/net 0.60.0: CVE-2026-78659, CVE-2026-78660, CVE-2026-78663,
# CVE-2026-78669, CVE-2026-97032.
# containerd/v2 2.3.6: CVE-2026-53495, CVE-2026-53493.
# swag/jsonutils 0.27.1: GHSA-xh24-9qpg-8w28.
# OpenTelemetry 1.45.0: CVE-2026-81870, exporter endpoint disclosure in
# verbose internal logs.
# Drop each override once the Trivy release includes that fixed version or newer.
RUN git clone --depth 1 --branch v${TRIVY_VERSION} https://github.com/aquasecurity/trivy.git /tmp/trivy \
 && cd /tmp/trivy \
 && go get google.golang.org/grpc@v1.83.2 \
      golang.org/x/crypto@v0.57.0 \
      golang.org/x/net@v0.60.0 \
      github.com/containerd/containerd/v2@v2.3.6 \
      github.com/go-openapi/swag/jsonutils@v0.27.1 \
      go.opentelemetry.io/otel/sdk@v1.45.0 \
      go.opentelemetry.io/otel/exporters/otlp/otlptrace@v1.45.0 \
      go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc@v1.45.0 \
 && go mod tidy \
 && CGO_ENABLED=0 GOEXPERIMENT=jsonv2 go build \
      -ldflags="-s -w -X=github.com/aquasecurity/trivy/pkg/version/app.ver=${TRIVY_VERSION}" \
      -o /out/trivy ./cmd/trivy \
 && test -x /out/trivy \
 && /out/trivy --version \
 && echo "--- linked Go module versions (verifying CVE-fix pins) ---" \
 && go version -m /out/trivy | grep -E '(stdlib|oras-go|rekor|golang\.org/x/(crypto|mod|net|text)|in-toto-golang|go-git/v5|go-billy/v5|containerd|grpc|buildkit|go-openapi/swag/jsonutils)' \
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
# Builders: Docker CLI plugins with patched Go + x/net
# =========================================
# Docker's packaged plugins still use Go 1.26.8 and x/net below 0.60.0.
# Rebuild them too; changing the ORAS/Trivy toolchain does not update these binaries.
FROM go-build AS buildx-build
ARG BUILDX_VERSION
# go-archive 0.3.0 fixes CVE-2026-17106; drop the override when Buildx includes it.
# The legacy Docker module is used only for random builder names. Use BuildKit's
# existing ID generator instead: new unnamed builders become builder-<id>;
# explicit and saved names are unchanged. Drop this patch once upstream removes
# the legacy dependency associated with CVE-2026-33997.
RUN git clone --depth 1 --branch v${BUILDX_VERSION} https://github.com/docker/buildx.git /tmp/buildx \
 && cd /tmp/buildx \
 && sed -i \
      -e 's|"github.com/docker/docker/pkg/namesgenerator"|"github.com/moby/buildkit/identity"|' \
      -e 's|for i := range 6 {|for range 6 {|' \
      -e 's|namesgenerator.GetRandomName(i)|"builder-" + identity.NewID()|' store/util.go \
 && grep -Fq '"builder-" + identity.NewID()' store/util.go \
 && go get golang.org/x/net@v0.60.0 golang.org/x/crypto@v0.57.0 github.com/moby/go-archive@v0.3.0 \
 && go mod tidy \
 && go mod vendor \
 && CGO_ENABLED=0 go build -mod=vendor -trimpath \
      -ldflags="-s -w -X github.com/docker/buildx/version.Version=v${BUILDX_VERSION} -X github.com/docker/buildx/version.Revision=$(git rev-parse HEAD).m -X github.com/docker/buildx/version.Package=github.com/docker/buildx" \
      -o /out/docker-buildx ./cmd/buildx \
 && go version -m /out/docker-buildx > /tmp/buildx-buildinfo.txt \
 && awk '$1 == "dep" && $2 == "github.com/docker/docker" { found=1 } END { exit found }' /tmp/buildx-buildinfo.txt \
 && /out/docker-buildx version \
 && mkdir -p /out/licenses/buildx \
 && cp LICENSE /out/licenses/buildx/ \
 && find vendor -type f \( -iname '*license*' -o -iname '*notice*' -o -iname '*copying*' -o -iname '*copyright*' \) \
      -exec cp --parents -t /out/licenses/buildx {} + \
 && rm -rf /tmp/buildx

FROM go-build AS compose-build
ARG COMPOSE_VERSION
# Match the upstream Linux release flags, including its e2e build tag.
RUN git clone --depth 1 --branch v${COMPOSE_VERSION} https://github.com/docker/compose.git /tmp/compose \
 && cd /tmp/compose \
 && go get golang.org/x/net@v0.60.0 golang.org/x/crypto@v0.57.0 \
 && go mod tidy \
 && go mod vendor \
 && CGO_ENABLED=0 go build -mod=vendor -trimpath -tags e2e \
      -ldflags="-w -X github.com/docker/compose/v5/internal.Version=v${COMPOSE_VERSION}" \
      -o /out/docker-compose ./cmd \
 && /out/docker-compose version \
 && mkdir -p /out/licenses/compose \
 && cp LICENSE NOTICE /out/licenses/compose/ \
 && find vendor -type f \( -iname '*license*' -o -iname '*notice*' -o -iname '*copying*' -o -iname '*copyright*' \) \
      -exec cp --parents -t /out/licenses/compose {} + \
 && rm -rf /tmp/compose

# =========================================
# Builders: preserve the runtime releases and Linux build features
# =========================================
FROM go-build AS runtime-build
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev make pkg-config libseccomp-dev \
 && rm -rf /var/lib/apt/lists/*

FROM runtime-build AS containerd-build
ARG CONTAINERD_VERSION
# A compiler update alone does not replace containerd's embedded x/net.
# Also fix CVE-2026-10722 in cilium/ebpf; retain the shim's upstream no_grpc build.
# Drop these overrides when containerd includes x/net 0.60.0 and ebpf 0.22.0.
RUN git clone --depth 1 --branch v${CONTAINERD_VERSION} https://github.com/containerd/containerd.git /tmp/containerd \
 && cd /tmp/containerd \
 && go get golang.org/x/net@v0.60.0 golang.org/x/crypto@v0.57.0 github.com/cilium/ebpf@v0.22.0 \
 && go mod tidy \
 && go mod vendor \
 && CGO_ENABLED=1 make GO_BUILD_FLAGS=-mod=vendor VERSION=v${CONTAINERD_VERSION} \
      bin/containerd bin/ctr bin/containerd-shim-runc-v2 \
 && mkdir -p /out/bin /out/licenses/containerd \
 && cp bin/containerd bin/ctr bin/containerd-shim-runc-v2 /out/bin/ \
 && cp LICENSE /out/licenses/containerd/ \
 && find . -maxdepth 1 -type f -iname '*notice*' -exec cp -t /out/licenses/containerd {} + \
 && find vendor -type f \( -iname '*license*' -o -iname '*notice*' -o -iname '*copying*' -o -iname '*copyright*' \) \
      -exec cp --parents -t /out/licenses/containerd {} + \
 && rm -rf /tmp/containerd

FROM runtime-build AS runc-build
ARG RUNC_VERSION
# moby/sys/user 0.4.1 fixes CVE-2026-61801. Keep seccomp and the packaged PIE mode.
# Refresh the other embedded modules too; drop overrides once runc carries them.
RUN git clone --depth 1 --branch v${RUNC_VERSION} https://github.com/opencontainers/runc.git /tmp/runc \
 && cd /tmp/runc \
 && go get golang.org/x/net@v0.60.0 golang.org/x/crypto@v0.57.0 \
      github.com/cilium/ebpf@v0.22.0 github.com/moby/sys/user@v0.4.1 \
 && go mod tidy \
 && go mod vendor \
 && make EXTRA_FLAGS=-mod=vendor BUILDTAGS="seccomp urfave_cli_no_docs" runc \
 && mkdir -p /out/bin /out/licenses/runc \
 && cp runc /out/bin/ \
 && cp LICENSE /out/licenses/runc/ \
 && find . -maxdepth 1 -type f -iname '*notice*' -exec cp -t /out/licenses/runc {} + \
 && find vendor -type f \( -iname '*license*' -o -iname '*notice*' -o -iname '*copying*' -o -iname '*copyright*' \) \
      -exec cp --parents -t /out/licenses/runc {} + \
 && rm -rf /tmp/runc

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
 && apt-get install -y --no-install-recommends docker-ce docker-ce-cli containerd.io libseccomp2 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=buildx-build /out/docker-buildx /usr/libexec/docker/cli-plugins/docker-buildx
COPY --from=compose-build /out/docker-compose /usr/libexec/docker/cli-plugins/docker-compose
COPY --from=containerd-build /out/bin/ /usr/bin/
COPY --from=runc-build /out/bin/runc /usr/bin/runc

# Retain upstream, dependency and Go runtime notices for the rebuilt binaries.
COPY --from=buildx-build /out/licenses/buildx /usr/share/doc/docker-buildx-plugin/licenses
COPY --from=compose-build /out/licenses/compose /usr/share/doc/docker-compose-plugin/licenses
COPY --from=containerd-build /out/licenses/containerd /usr/share/doc/containerd.io/containerd-licenses
COPY --from=runc-build /out/licenses/runc /usr/share/doc/containerd.io/runc-licenses
COPY --from=go-build /usr/local/go/LICENSE /usr/share/doc/docker-tools/Go-LICENSE
COPY --from=go-build /usr/local/go/PATENTS /usr/share/doc/docker-tools/Go-PATENTS

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

# Drop pip's vendored-dependency SBOM. pip 26.2 ships pip/_vendor/bom.cdx.json (a CycloneDX
# file) that scanners ingest, surfacing pip's internal vendored setuptools (70.3.0,
# CVE-2025-47273) and msgpack (1.1.2) as image findings even though they're pip internals,
# not installed packages. The base image's older pip omits this file and scans clean;
# removing it here matches that. Revisit once pip vendors patched setuptools/msgpack.
RUN find "$AZUREML_CONDA_ENVIRONMENT_PATH" -path '*/pip/_vendor/bom.cdx.json' -delete
