#!/bin/bash

if [ "$(uname)" == "Darwin" ]; then
   curl --output miniconda_installer.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-MacOSX-x86_64.sh
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
   curl --output miniconda_installer.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
else
   echo"ERROR: unsupported OS. Only MAC and Linux supported."
   exit 2
fi

bash miniconda_installer.sh -b -f -p $HOME/miniconda

source $HOME/miniconda/etc/profile.d/conda.sh

conda activate

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt
