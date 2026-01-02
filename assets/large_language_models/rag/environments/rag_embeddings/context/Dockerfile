FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:{{latest-image-tag}}

# Set the shared environment path
ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/rag-embeddings

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

# Sentence Transformers CPU only install
RUN pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cpu && \
    pip install --no-cache-dir transformers && \
    pip install --no-cache-dir tqdm numpy scikit-learn scipy nltk sentencepiece pillow && \
    pip install --no-cache-dir sentence-transformers --no-deps

RUN set -eux; \
    apt-get update; \
    # Update system packages to fix PAM vulnerability (USN-7580-1)
    apt-get upgrade -y; \
    apt-get install -y --no-install-recommends \
    # utilities for keeping Debian and OpenJDK CA certificates in sync
    ca-certificates p11-kit wget \
    ; \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME /usr/local/openjdk-21
ENV PATH $JAVA_HOME/bin:$PATH

# Default to UTF-8 file.encoding
ENV LANG C.UTF-8

# https://jdk.java.net/
# >
# > Java Development Kit builds, from Oracle
# >
ENV JAVA_VERSION 21-ea+22

RUN set -eux; \
    \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
        'amd64') \
            downloadUrl='https://download.java.net/java/GA/jdk21/fd2272bbf8e04c3dbaee13770090416c/35/GPL/openjdk-21_linux-x64_bin.tar.gz'; \
			downloadSha256='a30c454a9bef8f46d5f1bf3122830014a8fbe7ac03b5f8729bc3add4b92a1d0a'; \
            ;; \
        *) echo >&2 "error: unsupported architecture: '$arch'"; exit 1 ;; \
    esac; \
    \
    savedAptMark="$(apt-mark showmanual)"; \
    \
    wget --progress=dot:giga -O openjdk.tgz "$downloadUrl"; \
    echo "$downloadSha256 *openjdk.tgz" | sha256sum --strict --check -; \
    \
    mkdir -p "$JAVA_HOME"; \
    tar --extract \
        --file openjdk.tgz \
        --directory "$JAVA_HOME" \
        --strip-components 1 \
        --no-same-owner \
    ; \
    rm openjdk.tgz*; \
    \
    apt-mark auto '.*' > /dev/null; \
    [ -z "$savedAptMark" ] || apt-mark manual $savedAptMark > /dev/null; \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=true; \
    \
    # update "cacerts" bundle to use Debian's CA certificates (and make sure it stays up-to-date with changes to Debian's store)
    # see https://github.com/docker-library/openjdk/issues/327
    #     http://rabexc.org/posts/certificates-not-working-java#comment-4099504075
    #     https://salsa.debian.org/java-team/ca-certificates-java/blob/3e51a84e9104823319abeb31f880580e46f45a98/debian/jks-keystore.hook.in
    #     https://git.alpinelinux.org/aports/tree/community/java-cacerts/APKBUILD?id=761af65f38b4570093461e6546dcf6b179d2b624#n29
    mkdir -p /etc/ca-certificates/update.d; \
    ls -al /etc/ca-certificates; \
    { \
        echo '#!/usr/bin/env bash'; \
        echo 'set -Eeuo pipefail'; \
        echo 'trust extract --overwrite --format=java-cacerts --filter=ca-anchors --purpose=server-auth "$JAVA_HOME/lib/security/cacerts"'; \
    } > /etc/ca-certificates/update.d/docker-openjdk; \
    chmod +x /etc/ca-certificates/update.d/docker-openjdk; \
    /etc/ca-certificates/update.d/docker-openjdk; \
    update-ca-certificates; \
    \
    # https://github.com/docker-library/openjdk/issues/331#issuecomment-498834472
    find "$JAVA_HOME/lib" -name '*.so' -exec dirname '{}' ';' | sort -u > /etc/ld.so.conf.d/docker-openjdk.conf; \
    ldconfig; \
    \
    # https://github.com/docker-library/openjdk/issues/212#issuecomment-420979840
    # https://openjdk.java.net/jeps/341
    java -Xshare:dump; \
    \
    # basic smoke test
    fileEncoding="$(echo 'System.out.println(System.getProperty("file.encoding"))' | jshell -s -)"; [ "$fileEncoding" = 'UTF-8' ]; rm -rf ~/.java; \
    javac --version; \
    java --version

# Download nltk punkt and averaged_perceptron_trigger files to image for managed vnet support
RUN python3 -m nltk.downloader punkt && \
    python3 -m nltk.downloader averaged_perceptron_tagger

ENV TIKA_SERVER_JAR file:///tika-server.jar

# Install tika server
RUN downloadUrl='https://repo1.maven.org/maven2/org/apache/tika/tika-server-standard/3.2.2/tika-server-standard-3.2.2.jar'; \
    downloadMd5='https://repo1.maven.org/maven2/org/apache/tika/tika-server-standard/3.2.2/tika-server-standard-3.2.2.jar.md5'; \
    wget --progress=dot:giga -O tika-server.jar "$downloadUrl"; \
    # tika-python looks for tika-server.jar.md5 file along with TIKA_SERVER_JAR
    wget -O tika-server.jar.md5 "$downloadMd5"; \
    # basic smoke test
    python -c 'from tika import parser; parser.from_file("/root/.bashrc")'