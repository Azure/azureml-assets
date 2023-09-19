FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest

USER root:root

# Install Java
RUN mkdir -p /usr/share/man/man1
RUN apt-get update -y && \
    apt-get install -y openjdk-8-jdk libfuse-dev
ENV JAVA_HOME='/usr/lib/jvm/java-8-openjdk-amd64'

# Install Apache Spark
RUN wget -q https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz && \
   tar xzf spark-3.1.2-bin-hadoop3.2.tgz -C /opt && \
   mv /opt/spark-3.1.2-bin-hadoop3.2 /opt/spark && \
   rm spark-3.1.2-bin-hadoop3.2.tgz
ENV SPARK_HOME=/opt/spark
ENV PYSPARK_PYTHON=python
ENV PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python

# Install additional packages
WORKDIR /
COPY requirements.txt .
RUN pip install -r requirements.txt && rm requirements.txt

# Set the entrypoint script
RUN mkdir /scripts
COPY entrypoint.sh /scripts/
RUN chmod 755 /scripts/entrypoint.sh
ENTRYPOINT ["/scripts/entrypoint.sh"]