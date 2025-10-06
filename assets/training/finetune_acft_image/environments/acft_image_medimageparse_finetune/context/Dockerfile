FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch28x:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update

# --- Security Vulnerability Fixes ---
# Fix dpkg vulnerability (USN-7768-1)
RUN apt-get install -y --only-upgrade \
    dpkg-dev=1.21.1ubuntu2.6 \
    dpkg=1.21.1ubuntu2.6 \
    libdpkg-perl=1.21.1ubuntu2.6

# Fix GNU C Library vulnerability (USN-7760-1)
RUN apt-get install -y --only-upgrade \
    libc-bin=2.35-0ubuntu3.11 \
    libc6-dev=2.35-0ubuntu3.11 \
    libc-dev-bin=2.35-0ubuntu3.11 \
    locales=2.35-0ubuntu3.11 \
    libc6=2.35-0ubuntu3.11

# Fix libxml2 vulnerability (USN-7743-1)
RUN apt-get install -y --only-upgrade \
    libxml2=2.9.13+dfsg-1ubuntu0.9

COPY requirements.txt .

RUN pip install -r requirements.txt