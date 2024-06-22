#!/bin/bash

# Directory containing the input files
input_directory="/home/arjavp/Desktop/icarus/ground_truth_sims/sim-workdir-1"

# Get a list of input files in the directory
input_files=($(ls ${input_directory}/*.i))

# Loop through each input file
for input_file in "${input_files[@]}"
do
  # Extract the filename from the path
  filename=$(basename "$input_file")
  
  # Extract the base name without the extension
  base_name="${filename%.*}"
  
  # Create a new directory for the current input file
  output_directory="mooseherder_examples_${base_name}"
  mkdir -p "$output_directory"
  
  # Update the base directory in the Python script
  sed -i "s|mooseherder_examples/|${output_directory}/|" dataset_generator.py
  
  # Run the Python script with the current input file
  python dataset_generator.py "$filename"
  
  # Restore the original base directory in the Python script
  sed -i "s|${output_directory}/|mooseherder_examples/|" dataset_generator.py
done