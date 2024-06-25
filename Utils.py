import os
import time
import torch
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

output_folder = 'saved_data/cifar_test100'
count = count_txt_files(output_folder)
print(f"There are {count} .txt files in the folder.")
# count = count_png_files(output_folder)
# print(f"There are {count} .png files in the folder.")

### check number of files and process when meet condition

def wait_for_files(directory, target_count=40):
    """
    Continuously checks the directory until it contains at least target_count .txt files.

    Args:
    directory (str): The path to the directory to check.
    target_count (int): The minimum number of .txt files desired in the directory.
    """
    while True:
        count = count_txt_files(directory)
        print(f"Checking... There are currently {count} .txt files.")
        if count >= target_count:
            print(f"Reached target of {target_count} .txt files.")
            time.sleep(120)
            break
        time.sleep(20)  # Wait for 10 seconds before checking again

# # Example usage
# directory_path = 'saved_data/sd_turbo_500images_llava/'
# wait_for_files(directory_path)

### GPUs

# torch.cuda.set_device(2)
# if torch.cuda.is_available():
#     current_gpu = torch.cuda.current_device()
#     print(f"Current default GPU index: {current_gpu}")
#     print(f"Current default GPU name: {torch.cuda.get_device_name(current_gpu)}")
# else:
#     print("No GPUs available.")

def show_single(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_images_side_by_side(img_list):
    # Create a figure to contain the plots
    plt.figure(figsize=(15, 5))  # Increase the figure size as needed
    
    # Loop through the list of images
    for i, img in enumerate(img_list, 1):  # Start enumeration at 1
        npimg = img.numpy()  # Convert to numpy array if it's a tensor
        ax = plt.subplot(1, len(img_list), i)  # Create a subplot for each image
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Transpose dimensions
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        
    plt.show() 


def display_images_from_classes(folder_path, classes, num_images=9, grid_size=(3, 3)):
    """
    Display a grid of images sampled from specific classes in a folder.

    Args:
    folder_path (str): Path to the folder containing images.
    classes (list): List of class names to include in the sampling.
    num_images (int): Number of images to sample and display.
    grid_size (tuple): Dimensions of the grid (rows, columns) for displaying images.
    """
    # Ensure the folder exists and is a directory
    if not os.path.isdir(folder_path):
        print("The specified path is not a valid directory.")
        return

    # Get all files in the directory and filter for image files that start with any of the specified class names
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    filtered_files = [f for f in all_files if any(f.startswith(cls) for cls in classes) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # Randomly select files
    if len(filtered_files) < num_images:
        print(f"Not enough image files to sample from. Found only {len(filtered_files)} files.")
        return
    random.seed(42)
    sampled_files = random.sample(filtered_files, num_images)

    # Plot images in a grid
    fig, axes = plt.subplots(nrows=grid_size[0], ncols=grid_size[1], figsize=(15, 15))
    fig.suptitle('Sampled Images from Specific Classes', fontsize=16)

    for ax, image_file in zip(axes.flat, sampled_files):
        image_path = os.path.join(folder_path, image_file)
        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.axis('off')  # Hide axes
        ax.set_title(image_file.split('_')[0], fontsize=10, y=-0.15)  # Display the class part of the filename

    plt.tight_layout(pad=3.0)  # Adjust layout to make room for class labels
    plt.show()


def main():
    print("main.")

if __name__ == "__main__":
    main()