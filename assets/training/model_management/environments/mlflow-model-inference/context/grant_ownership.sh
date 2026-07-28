#!/bin/sh

# we transfer ownership of certain files and directories to dockeruser
# for each command below, the comment above it the error that occurs if we don't transfer ownership

# runsv iot-server: fatal: unable to open supervise/lock: file does not exist
# runsv gunicorn: fatal: unable to open supervise/lock: file does not exist
# runsv rsyslog: fatal: unable to open supervise/lock: file does not exist
# runsv nginx: fatal: unable to open supervise/lock: file does not exist
chown -R dockeruser /var/runit

# nginx: [alert] could not open error log file: open() "/var/log/nginx/error.log" failed (13: Permission denied)
chown -R dockeruser /var/log

# nginx: [emerg] mkdir() "/var/lib/nginx/body" failed (13: Permission denied)
chown -R dockeruser /var/lib/nginx

# nginx: [emerg] open() "/run/nginx.pid" failed (13: Permission denied)
# see nginx.conf: nginx writes its pid to /var/run/nginx.pid
chown -R dockeruser /run

# dockeruser needs execute permission for start_logger.sh
chmod +x /var/azureml-logger/start_logger.sh

# NotWritableError: The current user does not have write permissions to a required path.
# path: /opt/miniconda/pkgs/urls.txt
mkdir -p '/opt/miniconda/'
chown -R dockeruser /opt/miniconda/

# Permission error during Model.package() with userManagedDependencies = true
# mkdir: cannot create directory ‘/var/azureml-app’: Permission denied
# The command '/bin/sh -c mkdir -p '/var/azureml-app' && /var/azureml-util/download_asset.sh 'https://adbaws2171295715.blob.core.windows.net/azureml/LocalUpload/c37eefc7/tmpgrflit_r.py?sv=2019-02-02&sr=b&sig=JPoMtgp9tcsp7YiB209dR5zDrVduRZ2lKDlSzjRCGB4%3D&st=2021-03-06T01%3A10%3A50Z&se=2021-03-06T09%3A20%3A50Z&sp=r' '/var/azureml-app/tmpgrflit_r.py' && /var/azureml-util/download_asset.sh 'https://adbaws2171295715.blob.core.windows.net/azureml/LocalUpload/456c032d/score.py?sv=2019-02-02&sr=b&sig=LgMt879pdVmV33FjFb8IGjxO7NRZKaVtzj06HjiYEhw%3D&st=2021-03-06T01%3A10%3A51Z&se=2021-03-06T09%3A20%3A51Z&sp=r' '/var/azureml-app/score.py'' returned a non-zero code: 1
# 2021/03/06 01:22:30 Container failed during run: acb_step_1. No retries remaining.
# failed to run step ID: acb_step_1: exit status 1
# Explanation: EMS tries to create this directory as non-root and that fails.
# We create it ahead early to avoid the failure.
mkdir -p '/var/azureml-app'
chown -R dockeruser /var/azureml-app
