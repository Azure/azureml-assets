# openmpi image
FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

USER root

# sudo is expected by Singularity inside the image
# Fix OS-level security vulnerabilities:
# glibc (USN-8005-1), Python (USN-8018-1), GLib (USN-8017-1),
# nginx (USN-8038-1), GnuTLS (USN-8043-1), Expat (USN-8022-2),
# curl (USN-8062-1)
RUN apt-get update && ACCEPT_EULA=Y apt-get -y upgrade && \
    apt-get install -y sudo libxml2 sqlite3 libc-bin libc-dev locales libc6 dpkg-dev dpkg libdpkg-perl libssl-dev libssl3 openssl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# pip==26.0: CVE-2026-1703 path traversal during wheel extraction (GHSA-6vgw-5pg2-w6jp)
RUN pip install --upgrade pip==26.0

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

# temp fix for vulnerabilities, remove after pypi release
# wheel==0.46.2: CVE-2026-24049 path traversal (GHSA-8rrh-rw8j-w5fx)
# cryptography==46.0.5: CVE-2026-26007 EC subgroup validation (GHSA-r6ph-v2qm-q3c2)
RUN pip install --no-cache-dir scikit-learn==1.5.1 h2==4.3.0 urllib3==2.6.3 aiohttp==3.13.3 PyNaCl==1.6.2 wheel==0.46.2 cryptography==46.0.5 && rm -rf /root/.cache/pip

# The below file is required for baking the code into the environment 
COPY data_import_run.py /azureml/data_import/run.py

# dummy number to change when needing to force rebuild without changing the definition: 3

