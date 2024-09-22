#!/bin/bash

# Array of subfolders from 'i' to 'k'
folders=(k)

# Loop through each folder
for folder in "${folders[@]}"
do
    # Loop through numbers 1 to 4
    for i in {1..4}
    do
        # Run the Python script with the current config file from the subfolder
        python main.py -o ./app/config/lda/0922/$folder/lda_$i.json

        # Wait for 2 seconds before the next iteration
        sleep 2
    done
done