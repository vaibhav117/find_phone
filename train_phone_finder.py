import glob
import os
import numpy as np 
import cv2
import sys
import random
import math
import re
import time
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage.draw as draw

from phone_config import PhoneConfig
from dataset import get_dataset
import constants

from mrcnn import utils
import mrcnn.model as modellib

COCO_MODEL_PATH = os.path.join(constants.PRETRAINED_MODEL_PATH,"mask_rcnn_coco.h5")

def load_model(config):
    model = modellib.MaskRCNN(mode="training", config=config,model_dir=constants.MODEL_DIR)
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    return model

if __name__ == '__main__':
    file_path = sys.argv[1]
    dataset_train , dataset_val = get_dataset(file_path)
    config = PhoneConfig() 

    if not os.path.exists(COCO_MODEL_PATH):
        os.makedirs(constants.PRETRAINED_MODEL_PATH, exist_ok=False)
        utils.download_trained_weights(COCO_MODEL_PATH)
    
    model = load_model(config)

    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')