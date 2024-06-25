import os

def list_files_starting_with(folder_path, prefix):
    """ Check if the provided folder path exists, if exist print number of files that 
     meet the condition """
    if not os.path.isdir(folder_path):
        print(f"The folder path '{folder_path}' does not exist.")
        return

    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out files starting with the specified prefix
    matching_files = [file for file in files if file.startswith(prefix)]
    
    # Print the matching files
    if matching_files:
        print(len(matching_files))
    else:
        print(f"No files found starting with '{prefix}'.")
        
def count_txt_files(directory):
    """
    Counts the number of .txt files in the specified directory.

    Args:
    directory (str): The path to the directory to search for .txt files.

    Returns:
    int: The number of .txt files in the directory.
    """
    txt_count = 0
    # List all files and directories in the specified directory
    for entry in os.listdir(directory):
        # Construct full path
        full_path = os.path.join(directory, entry)
        # Check if it's a file with a .txt extension
        if os.path.isfile(full_path) and entry.endswith('.txt'):
            txt_count += 1
    
    return txt_count

# output_folder = 'saved_data/ImageNet/ImageNet_sdxl_llavaprompt_1300_3real_i2i'
# count = count_txt_files(output_folder)
# print(f"There are {count} .txt files in the folder.")
# count = count_png_files(output_folder)
# print(f"There are {count} .png files in the folder.")