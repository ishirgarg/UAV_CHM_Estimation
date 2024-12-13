'''
Copyright (c) 2024 Ishir Garg

Utils for standard computer vision operations
'''

import numpy as np

def box_prediction_to_xyxy(pred):
    '''Converts DeepForest outputs to xyxy format'''
    return np.array([int(pred["xmin"]), int(pred["ymin"]), int(pred["xmax"]), int(pred["ymax"])])