import os


def rename_files_sequentially(directory, prefix=''):
    """
    Renames all files in the specified directory to a sequential number with a prefix.
    Args:
    - directory: The path to the directory containing files to rename.
    - prefix: Optional prefix for file names.
    """
    # List all files in the directory, sorted alphabetically
    files = sorted([file for file in os.listdir(directory)
                   if os.path.isfile(os.path.join(directory, file))])

    # Loop through each file and rename it
    for idx, file_name in enumerate(files):
        # Create the new file name with sequential numbering
        extension = file_name.split('.')[-1]  # Get file extension
        new_name = f"{prefix}{idx}.{extension}"  # Format new name

        # Build full file paths
        old_path = os.path.join(directory, file_name)
        new_path = os.path.join(directory, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f'Renamed "{file_name}" to "{new_name}"')


# Specify the directory and optional prefix
directory_path = 'BU-BIL_Dataset2/GoldStandard/chian1'
rename_files_sequentially(directory_path, prefix='')

print("Files have been renamed successfully.")
