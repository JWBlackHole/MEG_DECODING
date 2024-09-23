#!/bin/bash

# Loop through numbers 1 to 4 for subfolder 'l'
for i in {1..4}
do
    # Run the Python script with the current config file from the subfolder 'l'
    python main.py -o ./app/config/lda/0922/l/lda_$i.json

    # Wait for 2 seconds before the next iteration
    sleep 2
done

# Loop through numbers 1 to 3 for subfolder 'o'
for i in {2..4}
do
    # Run the Python script with the current config file from the subfolder 'o'
    python main.py -o ./app/config/lda/0922/o/lda_$i.json

    # Wait for 2 seconds before the next iteration
    sleep 2
done