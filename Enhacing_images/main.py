import cv2
import os
import shutil

# Function to compute image sharpness using the variance of Laplacian
def compute_sharpness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance

# Paths
input_folder = 'shapes_rotation/images'
output_folder = 'output1/images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Compute sharpness for each image and store in a list
sharpness_scores = []
for image_file in os.listdir(input_folder):
    if image_file.endswith('.png'):
        input_path = os.path.join(input_folder, image_file)
        sharpness = compute_sharpness(input_path)
        sharpness_scores.append((image_file, sharpness))

# Sort images by sharpness in descending order and select the top 81
sharpness_scores.sort(key=lambda x: x[1], reverse=True)
best_images = [image_file for image_file, _ in sharpness_scores[:81]]

# Copy selected images to the output folder
for image_file in best_images:
    input_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, image_file)
    shutil.copy(input_path, output_path)

# Update events.txt and images.txt
with open('shapes_rotation/events.txt', 'r') as events_file:
    with open('output1/events.txt', 'w') as events_output:
        for line in events_file:
            if any(image_file.split('.')[0] in line for image_file in best_images):
                events_output.write(line)

with open('shapes_rotation/images.txt', 'r') as images_file:
    with open('output1/images.txt', 'w') as images_output:
        for line in images_file:
            if any(image_file in line for image_file in best_images):
                images_output.write(line)

# Copy calib.txt to the output folder
shutil.copy('shapes_rotation/calib.txt', 'output1/')
