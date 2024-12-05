#!/bin/bash

# Check if the input argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <i>"
    exit 1
fi

# Assign the first argument to the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=$1

# Confirm the setting
echo "CUDA_VISIBLE_DEVICES is set to $CUDA_VISIBLE_DEVICES"

