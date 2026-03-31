#!/bin/bash
# run from the project folder on the Pi (after chmod +x run.sh)
set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
# windowed debug on hdmi second monitor:  FILTER_WINDOWED=1 ./run.sh
exec python3 display.py "$@"
