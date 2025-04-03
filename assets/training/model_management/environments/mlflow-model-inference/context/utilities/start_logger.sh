#!/usr/bin/env bash

if ! pgrep -x "rsyslogd" > /dev/null; then
  exec rsyslogd -n
fi