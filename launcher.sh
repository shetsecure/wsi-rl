#!/bin/bash

# Store the path where JSON conf files are located
CONF_PATH="confs"

# Iterate over each conf file in the directory
for conf_file in $CONF_PATH/*.yaml; do

    # Extract just the file name (without path and extension)
    filename=$(basename -- "$conf_file")
    fname_no_ext="${filename%.*}"

    # Call the Slurm script with sbatch, setting output and error filenames

    sbatch --output="${fname_no_ext}%j.out" --error="${fname_no_ext}%j.out" --job-name="${fname_no_ext}" slurm_script.sh $conf_file

done
