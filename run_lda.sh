#!/bin/bash

# Loop through numbers 1 to 27
for i in {1..27}
do
    # Run the Python script with the current config file
    python main.py -o ./app/config/lda/lda_$i.json
    
    # Wait for 2 seconds before the next iteration
    sleep 2
done