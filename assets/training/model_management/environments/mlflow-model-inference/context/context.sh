#!/bin/bash

echo "Prepare files in /var/ ..."

mkdir -p /var/runit
mkdir -p /var/azureml-app
mkdir -p /opt/miniconda/
mkdir -p etc/nginx/sites-available

cp -R runit/gunicorn /var/runit/gunicorn/
cp -R runit/nginx /var/runit/nginx/
cp -R runit/rsyslog /var/runit/rsyslog/
cp -R common/aml_logger /var/azureml-logger
cp -R utilities/start_logger.sh /var/azureml-logger/start_logger.sh
cp -R configs/app etc/nginx/sites-available/app

sed -i 's/\r$//g' /var/runit/gunicorn/run
sed -i 's/\r$//g' /var/runit/gunicorn/finish
sed -i 's/\r$//g' /var/runit/nginx/run
sed -i 's/\r$//g' /var/runit/nginx/finish

chmod +x var/runit/*/*
chmod +x var/azureml-logger/start_logger.sh

echo "Done!"