#!/bin/bash

# Array of subfolders from 'a' to 'h'
folders=(a b c d f g h)

# Loop through each folder
for folder in "${folders[@]}"
do
    # Loop through numbers 1 to 27
    for i in {1..27}
    do
        # Run the Python script with the current config file from the subfolder
        python main.py -o ./app/config/lda/0921/$folder/lda_$i.json

        # Wait for 2 seconds before the next iteration
        sleep 2
    done
done
