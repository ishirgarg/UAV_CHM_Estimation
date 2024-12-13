'''
Copyright (c) 2024 Ishir Garg

Generates cropped images from the training set mosaics
'''

import numpy as np
from PIL import Image
from src.datasets.NeonDataset import NEONDataset


def find_annotations(x_tl: int, y_tl: int, x_br: int, y_br: int, annotations: np.ndarray, min_width: int):
    '''Finds the subset of annotated bounding boxes in the box defined by the given coordinates
    
    Params:
        x_tl: x-coord of top left corner
        y_tl: x-coord of top left corner
        x_br: x-coord of bottom right corner
        y_br: y-coord of bottom right corner
        annotations: List of annotated bounding boxes for full image
        min_width: Minimum width of a bounding box to be included in the subset
    '''
    indices = []
    for i, ann in enumerate(annotations):
        box_x_tl = max(x_tl, ann[0])
        box_y_tl = max(y_tl, ann[1])
        box_x_br = min(x_br, ann[2])
        box_y_br = min(y_br, ann[3])
        if box_x_tl + min_width - 1 <= box_x_br and box_y_tl + min_width - 1 <= box_y_br:
            indices.append(i)
    return annotations[indices]


def generate_image_crops(image: dict, size: tuple[int, int], overlap: tuple[int, int], min_width: int):
    '''Generates crops of given size from an rgb image
    image: dict with keys 'rgb' and 'annotation'
    '''
    annotation = image["annotation"]
    chm = image["chm"]
    rgb = image["rgb"]

    crops_arr = []
    for i in range(0, rgb.shape[0] - size[0] + 1, size[0] - overlap[0]):
        for j in range(0, rgb.shape[1] - size[1] + 1, size[1] - overlap[1]):
            crops_arr.append({'rgb': rgb[i:i+size[0], j:j+size[1]] / 256,
                              'chm': chm[i//10:(i+size[0])//10, j//10:(j+size[1])//10]})

    return np.array(crops_arr)


def generate_dataset_crops(images: np.ndarray, size: tuple[int, int], overlap: tuple[int, int], min_width: int, output_path: str):
    '''Takes an array of images and annotations and generates crops for all of them with annotations'''
    cropped_images = [generate_image_crops(img, size, overlap, min_width) for img in images]
    # Flatten into list of images
    cropped_images = [img for sublist in cropped_images for img in sublist]
    np.save(output_path, cropped_images)