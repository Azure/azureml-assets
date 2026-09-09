FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

# Upgrade every inherited Ubuntu package reported by the image scan and fail the
# build if an installed package does not meet its advisory's fixed-version floor.
RUN set -eux; \
    packages='libssl3t64 openssl perl libperl5.38t64 perl-modules-5.38 perl-base libp11-kit0 diffutils coreutils curl libcurl3t64-gnutls libcurl4t64 libattr1 libssh-4 zlib1g libgcrypt20 libsystemd-shared systemd libudev1 systemd-resolved systemd-timesyncd libsystemd0 systemd-dev nginx-light nginx nginx-common'; \
    held_packages="$(apt-mark showhold)"; \
    for package in ${packages}; do \
        if printf '%s\n' "${held_packages}" | grep -Fxq "${package}"; then \
            echo "Security update blocked by held package: ${package}" >&2; \
            exit 1; \
        fi; \
    done; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade --no-install-recommends ${packages}; \
    check_version() { \
        package="$1"; \
        required="$2"; \
        if dpkg-query -W -f='${db:Status-Abbrev}' "${package}" 2>/dev/null | grep -q '^ii '; then \
            installed="$(dpkg-query -W -f='${Version}' "${package}")"; \
            dpkg --compare-versions "${installed}" ge "${required}" || { \
                echo "${package} ${installed} is below required version ${required}" >&2; \
                exit 1; \
            }; \
        fi; \
    }; \
    check_version libssl3t64 3.0.13-0ubuntu3.15; \
    check_version openssl 3.0.13-0ubuntu3.15; \
    check_version perl 5.38.2-3.2ubuntu0.4; \
    check_version libperl5.38t64 5.38.2-3.2ubuntu0.4; \
    check_version perl-modules-5.38 5.38.2-3.2ubuntu0.4; \
    check_version perl-base 5.38.2-3.2ubuntu0.4; \
    check_version libp11-kit0 0.25.3-4ubuntu2.2; \
    check_version diffutils 1:3.10-1ubuntu0.1; \
    check_version coreutils 9.4-3ubuntu6.3; \
    check_version curl 8.5.0-2ubuntu10.13; \
    check_version libcurl3t64-gnutls 8.5.0-2ubuntu10.13; \
    check_version libcurl4t64 8.5.0-2ubuntu10.13; \
    check_version libattr1 1:2.5.2-1ubuntu0.1; \
    check_version libssh-4 0.10.6-2ubuntu0.5; \
    check_version zlib1g 1:1.3.dfsg-3.1ubuntu2.2; \
    check_version libgcrypt20 1.10.3-2ubuntu0.2; \
    check_version libsystemd-shared 255.4-1ubuntu8.17; \
    check_version systemd 255.4-1ubuntu8.17; \
    check_version libudev1 255.4-1ubuntu8.17; \
    check_version systemd-resolved 255.4-1ubuntu8.17; \
    check_version systemd-timesyncd 255.4-1ubuntu8.17; \
    check_version libsystemd0 255.4-1ubuntu8.17; \
    check_version systemd-dev 255.4-1ubuntu8.17; \
    check_version nginx-light 1.24.0-2ubuntu7.16; \
    check_version nginx 1.24.0-2ubuntu7.16; \
    check_version nginx-common 1.24.0-2ubuntu7.16; \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade 'pip==26.2.1' && \
    python -m pip install --no-cache-dir -r /app/requirements.txt

# Security: starlette is pulled in transitively (via fastapi). Bump to >=1.3.1 to fix
# GHSA-82w8-qh3p-5jfq and GHSA-wqp7-x3pw-xc5r (both HIGH; fixed in 1.3.1). Applied as a
# separate post-install upgrade rather than a requirements.txt pin because forcing the
# floor inside the combined resolution triggers pip "resolution-too-deep". (ICM 831170435)
RUN python -m pip install --no-cache-dir --upgrade 'starlette>=1.3.1'

# Upgrade vulnerable transitive dependencies after the primary dependency graph resolves.
# azureml-mlflow currently caps cryptography below 49; this security floor intentionally
# overrides that stale metadata until the upstream constraint includes cryptography 50.
RUN python -m pip install --no-cache-dir --upgrade \
        'pip==26.2.1' \
        'cryptography==50.0.1' \
        'h2==4.4.1' \
        'nltk==3.10.3' \
        'sqlparse==0.6.0'

# Copy your Python file into the image
COPY evaluate_on_data.py /app/evaluate_on_data.py
COPY model_target.py /app/model_target.py
COPY save_evaluation.py /app/save_evaluation.py
ADD online_eval /app/online_eval