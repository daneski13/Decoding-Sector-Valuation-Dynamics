#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate Decoding-Sector-Valuation-Dynamics

# Find all .ipynb files with the pattern "Thesis - " and execute them
find . -maxdepth 1 -type f -name "Thesis - *.ipynb" | while read -r notebook; do
    echo "Executing notebook: $notebook"
    jupyter nbconvert --to notebook --execute "$notebook" --inplace --ExecutePreprocessor.kernel_name=Decoding-Sector-Valuation-Dynamics
    echo "Execution complete for: $notebook"
done
