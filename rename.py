import os
from pathlib import Path

'''
Renames the ground truth files into a more human-readable format 
'''

# Specify the directory containing your files
directory = Path(Path.cwd(), 'ground_truth_sims')

# List all files in the directory
files = sorted(os.listdir(directory))

# Rename each file with .e extension
for count, filename in enumerate(files, start=1):
    if filename.endswith('.e'):
        # Construct the new name
        new_name = f"ground_truth_{count}.e"
        
        # Get full paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_name)
        
        # Rename the file
        os.rename(old_file, new_file)
