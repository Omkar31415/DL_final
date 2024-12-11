import os
import re

def correct_timestamps(data_folder):
    # Read the existing image_timestamps.txt file
    timestamps = []
    with open(os.path.join(data_folder, 'images_timestamps.txt'), 'r') as f:
        for line in f:
            timestamps.append(int(line.strip()))

    # Read the image filenames from the images folder
    image_files = sorted(os.listdir(os.path.join(data_folder, 'images')))

    # Create a new file for corrected timestamps
    with open(os.path.join(data_folder, 'image_timestamps_corrected.txt'), 'w') as f:
        for i, image_file in enumerate(image_files):
            # Extract the frame number from the image filename
            frame_number = re.search(r'frame_(\d+)\.png', image_file).group(1)
            # Write the corrected format to the new file with .png appended
            f.write(f'images/frame_{frame_number.zfill(10)}.png {timestamps[i]}\n')

# Example usage
data_folder = 'data/boxes_6dof/'
correct_timestamps(data_folder)