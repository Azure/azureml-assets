#!/usr/bin/env bash

if [[ -z "${AZUREML_CONDA_ENVIRONMENT_PATH}" ]]; then
  action_binary="$(conda info --root)/bin/python ${AML_LOGGER_ROOT:-/var/azureml-logger}/rsyslog_plugin.py"
else
  action_binary="$AZUREML_CONDA_ENVIRONMENT_PATH/bin/python ${AML_LOGGER_ROOT:-/var/azureml-logger}/rsyslog_plugin.py"
fi

sed -i "s|<action binary placeholder>|${action_binary}|g" /etc/rsyslog.conf

if ! pgrep -x "rsyslogd" > /dev/null; then
  exec rsyslogd -n
fi