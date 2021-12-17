import sys
import cv2

import mrcnn.model as modellib
from mrcnn import utils

from phone_config import InferenceConfig
from dataset import get_dataset
import constants


def load_image(file_path, config):
    image = cv2.imread(file_path)
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    return image

if __name__ == '__main__':
    file_path = sys.argv[1]
    inference_config = InferenceConfig()
    print(f"\n\n\n{inference_config}\n\n\n")
    img = load_image(file_path,config=inference_config)

       
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=constants.MODEL_DIR)
    model_path = model.find_last()
    model.load_weights(model_path, by_name=True)

    prediction = model.detect([original_image], verbose=0)[0]
    print(prediction)