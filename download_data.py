import os
import json
from cryoet_data_portal import Client, Tomogram, Run, Dataset, Annotation
import mrcfile
import matplotlib.pyplot as plt
import numpy as np
import imageio


def download_dataset(dest_path, dataset_id, dataset_name):
    print('Downloading the dataset...')
    # Instantiate a client, using the data portal GraphQL API by default
    client = Client()

    # Find all tomograms and annotations with a dataset id and dataset name
    tomos = Tomogram.find(client, [Tomogram.tomogram_voxel_spacing.run.dataset.id == dataset_id, Tomogram.tomogram_voxel_spacing.run.name == dataset_name])
    annotations = Annotation.find(client, [Annotation.tomogram_voxel_spacing.run.dataset.id == dataset_id, Annotation.tomogram_voxel_spacing.run.name == dataset_name])

    for tomo,anno in zip(tomos, annotations):
        print(tomo.name)
        
        # Download a tomogram in the MRC format
        # The dest path needs to be a file name, not a directory
        tomo.download_mrcfile(dest_path=dest_path)
        # Download annotations
        breakpoint()
        anno.download(dest_path=dest_path+'_annotations')

def visualize_tomogram(dest_path, dataset_id, dataset_name, gif_spacing):
    with mrcfile.open(dest_path) as mrc:
        print(mrc.data.shape)
        z, x, y = mrc.data.shape
        vmin = np.percentile(mrc.data, 5)
        vmax = np.percentile(mrc.data, 95)
        plt.figure()
        plt.imshow(mrc.data[z//2, :, :], vmin = vmin, vmax = vmax, cmap = 'gray')
        plt.savefig('../output-cryo-segment/dataset_' + str(dataset_id) + '_' + str(dataset_name) + '.png')

        # Create gif for the tomogram
        

        images = []
        for i in range(0,z,gif_spacing):
            image_i = np.copy(mrc.data[i, :, :])
            image_i[image_i < vmin] = vmin
            image_i[image_i > vmax] = vmax
            image_i = (image_i - vmin) / (vmax - vmin)
            images.append(image_i*255)
        imageio.mimsave('../output-cryo-segment/dataset_' + str(dataset_id) + '_' + str(dataset_name) + '.gif', images)


if __name__ == "__main__":
    # Inputs
    dataset_id = 10006
    dataset_name = 'TS_004'
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