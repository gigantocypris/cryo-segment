import os
import json
from cryoet_data_portal import Client, Tomogram, Run, Dataset, Annotation
import mrcfile
import matplotlib.pyplot as plt
import numpy as np
import imageio


def download_dataset(dest_path, dataset_id, dataset_name, download_annotations=True):
    print('Downloading the dataset...')
    # Instantiate a client, using the data portal GraphQL API by default
    client = Client()

    # Find all tomograms and annotations with a dataset id and dataset name
    tomos = Tomogram.find(client, [Tomogram.tomogram_voxel_spacing.run.dataset.id == dataset_id, Tomogram.tomogram_voxel_spacing.run.name == dataset_name])
    
    for tomo in tomos:
        print('Tomogram name: ' + tomo.name)
        # Download a tomogram in the MRC format
        # The dest path needs to be a file name, not a directory
        tomo.download_mrcfile(dest_path=dest_path)

    if download_annotations:
        annotations = Annotation.find(client, [Annotation.tomogram_voxel_spacing.run.dataset.id == dataset_id, Annotation.tomogram_voxel_spacing.run.name == dataset_name])

        for ind, anno in enumerate(annotations):
            print(anno)
            anno.download(dest_path=dest_path+'_annotations_' + str(ind))

def visualize_tomogram(dest_path, dataset_id, dataset_name, gif_spacing, annotation=False, dest_path_suffix=''):

    dest_path = dest_path + dest_path_suffix

    with mrcfile.open(dest_path) as mrc:
        print(mrc.data.shape)
        z, x, y = mrc.data.shape
        if annotation:
            vmin = mrc.data.min()
            vmax = mrc.data.max()
        else:
            vmin = np.percentile(mrc.data, 5)
            vmax = np.percentile(mrc.data, 95)
        plt.figure()
        plt.imshow(mrc.data[z//2, :, :], vmin = vmin, vmax = vmax, cmap = 'gray')
        plt.savefig('../output-cryo-segment/dataset_' + str(dataset_id) + '_' + str(dataset_name) + dest_path_suffix + '.png')

        # Create gif for the tomogram
        
        images = []
        for i in range(0,z,gif_spacing):
            image_i = np.copy(mrc.data[i, :, :])
            if annotation:
                pass
            else:
                image_i[image_i < vmin] = vmin
                image_i[image_i > vmax] = vmax
            image_i = (image_i - vmin) / (vmax - vmin)
            images.append(image_i*255)
        imageio.mimsave('../output-cryo-segment/dataset_' + str(dataset_id) + '_' + str(dataset_name) + dest_path_suffix + '.gif', images)
    return images

def create_annotated_gif(images, annotations, dataset_id, dataset_name, dest_path_suffix, save_single_images=False):
    annotated_images = []
    for ind, (image, annotation) in enumerate(zip(images, annotations)):
        # annotation = annotation*3//255
        annotation = np.flipud(annotation*3//255) # np.flipud needed for TE13
        breakpoint()
        red_image = np.copy(image)
        green_image = np.copy(image)
        blue_image = np.copy(image)
        for color in range(1,4):
            red_image[annotation==color] = 255*(color==1)
            green_image[annotation==color] = 255*(color==2)
            blue_image[annotation==color] = 255*(color==3)
        annotated_image = np.stack([red_image, green_image, blue_image], axis=2)
        annotated_image = annotated_image.astype(np.uint8)
        annotated_images.append(annotated_image)
        if ind == len(images)//2:
            imageio.imsave('../output-cryo-segment/dataset_' + str(dataset_id) + '_' + str(dataset_name) + dest_path_suffix + '.png', annotated_image)
    annotated_images = np.stack(annotated_images,axis=0)
    imageio.mimsave('../output-cryo-segment/dataset_' + str(dataset_id) + '_' + str(dataset_name) + dest_path_suffix + '.gif', annotated_images)

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