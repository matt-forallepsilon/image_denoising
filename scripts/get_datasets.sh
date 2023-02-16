#!/bin/bash

if [-e "denoising_datasets_main.zip"]; then
    echo "file aleady exists" >&2
else
    curl -LJO https://github.com/majedelhelou/denoising_datasets/archive/refs/heads/main.zip
    unzip denoising_datasets-main.zip
fi
