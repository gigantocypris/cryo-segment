import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from download_data import download_dataset, visualize_tomogram, create_annotated_gif
# best dataset on cryo ET portal for membrane segmentation ground truth: https://cryoetdataportal.czscience.com/datasets/10010

# Inputs
dataset_id = 10010
dataset_name = 'TE13'
gif_spacing = 5

# Output path for the tomogram file (not a directory)
dest_path = '../output-cryo-segment/dataset_' + str(dataset_id) + '_' + str(dataset_name)

# check if the dataset has already been downloaded
if os.path.isfile(dest_path):
    print('The dataset has already been downloaded.')
else:
    download_dataset(dest_path, dataset_id, dataset_name)

images = visualize_tomogram(dest_path, dataset_id, dataset_name, gif_spacing)
annotations = visualize_tomogram(dest_path, dataset_id, dataset_name, gif_spacing, annotation=True, dest_path_suffix = '_annotations_0')
create_annotated_gif(images, annotations, dataset_id, dataset_name, dest_path_suffix = '_annotations_0_image')
