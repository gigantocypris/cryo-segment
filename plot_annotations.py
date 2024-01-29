import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from download_data import download_dataset, visualize_tomogram
# best dataset on cryo ET portal for membrane segmentation ground truth: https://cryoetdataportal.czscience.com/datasets/10010

# Inputs
dataset_id = 10010
dataset_name = 'TE2'
gif_spacing = 5

# Output path for the tomogram file (not a directory)
dest_path = '../output-cryo-segment/dataset_' + str(dataset_id) + '_' + str(dataset_name)

# check if the dataset has already been downloaded
if os.path.isfile(dest_path):
    print('The dataset has already been downloaded.')
else:
    print('Downloading the dataset...')
    download_dataset(dest_path, dataset_id, dataset_name)

visualize_tomogram(dest_path, dataset_id, dataset_name, gif_spacing)

breakpoint()
# Load the image
image = plt.imread('path_to_your_image.jpg')

# Parse the segmentation annotations from the file
with open('path_to_your_annotations.json', 'r') as f:
    annotations = json.load(f)

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the image
ax.imshow(image)

# Iterate through each annotation
for annotation in annotations:
    # Extract location and rotation matrix
    location = np.array([annotation['location']['x'], annotation['location']['y']])
    rotation_matrix = np.array(annotation['xyz_rotation_matrix'])
    
    # Example code to draw a polygon for each annotation
    # You need to implement the logic based on your annotation format
    # Here, we assume the annotation contains points of a polygon
    polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])  # Example polygon, replace with your annotation data
    transformed_polygon = np.dot(polygon, rotation_matrix.T) + location
    ax.add_patch(Polygon(transformed_polygon, alpha=0.5, closed=True, fill=True, edgecolor='r'))

# Show the image with segmentation annotations
plt.show()
