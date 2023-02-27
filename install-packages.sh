#!/bin/bash
set -e

sudo apt-get update && \
    sudo apt-get install -y build-essential graphviz

pip3 install -r pykilosort-requirements.txt
pip3 install -e '.[kilosort,cuda]'