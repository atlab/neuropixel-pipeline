#!/bin/bash
set -eo pipefail

apt-get update && \
    apt-get install -y build-essential graphviz

pip3 install -r pykilosort-requirements.txt
pip3 install -e '.[kilosort,cuda]'