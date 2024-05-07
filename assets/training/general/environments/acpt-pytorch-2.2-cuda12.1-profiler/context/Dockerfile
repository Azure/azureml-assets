FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu121-py310-torch22x:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

# Install pip dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN /var/requirements/install_system_requirements.sh && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

# support Deepspeed launcher requirement of passwordless ssh login
RUN apt-get update
RUN apt-get install -y openssh-server openssh-client


ENV DEBIAN_FRONTEND=noninteractive
RUN apt install -y curl tmux wget systemd

# install dcgm
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g') && wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update
RUN apt-get install -y datacenter-gpu-manager


# install dynolog
RUN wget https://github.com/facebookincubator/dynolog/releases/download/v0.2.2/dynolog_0.2.2-0-amd64.deb
RUN dpkg -i dynolog_0.2.2-0-amd64.deb
RUN echo "--enable_ipc_monitor" | tee -a /etc/dynolog.gflags
RUN echo "--enable_gpu_monitor" | tee -a /etc/dynolog.gflags
RUN echo "--dcgm_lib_path=/usr/lib/x86_64-linux-gnu/libdcgm.so" | tee -a /etc/dynolog.gflags
RUN touch /var/log/dynolog.log
RUN echo "--use_JSON" | tee -a /etc/dynolog.gflags

# run profiler
#dyno gputrace --log-file kineto-trace.json

# install HTA
RUN pip install HolisticTraceAnalysis
WORKDIR /workspace
ENV KINETO_USE_DAEMON=1

CMD ["sudo", "systemctl", "--now enable nvidia-dcgm"]

CMD ["sudo", "nv-hostengine", "-n --service-account nvidia-dcgm"]

# Run dynolog with sudo
CMD ["sudo", "dynolog", "--flagfile=/etc/dynolog.gflags"]
