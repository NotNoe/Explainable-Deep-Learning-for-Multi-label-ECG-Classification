import os
import shutil


# Unnecessary files
files_to_delete = ["LICENSE.txt", "SHA256SUMS.txt", "example_physionet.py", "ptbxl_v102_changelog.txt", "ptbxl_v103_changelog.txt"]
for file in files_to_delete:
    file_path = os.path.join("ptbxl", file)
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"File {file} has been deleted.")
    else:
        print(f"File {file} not found in ptbxl folder.")
if os.path.isdir('ptbxl/records100'):
    shutil.rmtree('ptbxl/records100')  # Eliminamos la carpeta records100
    print("Folder records100 has been deleted.")
else:
    print("Folder records100 not found in ptbxl folder.")
file_path = './ptbxl/RECORDS'
# We read the original file and modify its content
with open(file_path, 'r') as infile:
    lines = infile.readlines()

with open(file_path, 'w') as outfile:
    for line in lines:
        line = line.strip()  # We remove leading and trailing whitespace
        
        # If the line ends with _lr, we ignore it
        if line.endswith('_lr'):
            continue
        
        # If _lr is in the line but not at the end, we remove everything up to _lr
        if '_lr' in line:
            line = line.split('_lr', 1)[1]  # Split and take only the part after _lr
        
        # Write the modified line to the same file if it's not empty
        if line:
            outfile.write(line + '\n')

print("The file has been overwritten with the modifications.")
