'''
Copyright (c) 2024 Ishir Garg

Utils for plotting various image data
'''

import matplotlib.pyplot as plt
from matplotlib import patches


def visualize_rgb_chm(rgb_img, chm_img):
    '''Takes in a rgb and chm images and plots them side-by-side'''
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(rgb_img)
    ax[1].imshow(chm_img)
    plt.show()

def visualize_chms(chm1, chm2):
    min_val = min(chm1.min(), chm2.min())
    max_val = max(chm1.max(), chm2.max())
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(chm1, cmap='seismic', vmin=min_val, vmax=max_val)
    ax[1].imshow(chm2, cmap='seismic', vmin=min_val, vmax=max_val)
    plt.show()

def visualize_rgb_chm_ann(rgb_img, chm_img, annotations):
    '''Takes in a rgb and chm images and plots them side-by-side'''
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(rgb_img)
    ax[1].imshow(chm_img)
    for bbox in annotations:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=0.2, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
    plt.show()


def visualize(datapoint):
    '''Takes in a dict with keys 'rgb' and 'multi' and plots them side-by-side'''
    if "annotation" in datapoint:
        visualize_rgb_chm_ann(datapoint["rgb"], datapoint["chm"], datapoint["annotation"])
    else:
        visualize_rgb_chm(datapoint["rgb"], datapoint["chm"])
