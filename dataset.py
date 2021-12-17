import tensorflow as tf
import cv2
import numpy as np

from mrcnn import utils
import skimage.draw as draw

import constants
from phone_config import PhoneConfig

# Setting up Dataset for Phones 
class PhoneDataset(utils.Dataset):
    """Generates the dataset for phones for Mask RCNN
    """

    def load_shapes(self, mode, X, Y, imgs, img_paths):
    
        assert mode == "train" or  mode == "test" or mode == "val", "Mode can only be train, val or test"
        
        # Add classes
        self.add_class("phone", 1, "phone")
        self.class2index = {
            "phone": 1,
        }
        
        # Assigning class variables
        self.images = imgs
        self.img_paths = img_paths
        self.X = X
        self.Y = Y
        self.img_h = constants.IMAGE_HEIGHT # all images are 326 pixels in height
        self.img_w = constants.IMAGE_WIDTH  # all images are 326 pixels in width
        self.bboxSize = constants.BBOXSIZE # as described above 0.02 is a good bounding box size

        # Dividing dataset into training and validation set.
        train_start = 0
        train_end = round(100*constants.TRAIN_DATA_SPLIT_RATIO - 1) # 80% split

        val_start = train_end+1
        val_end = round(val_start + 100*constants.VALUATION_DATA_SPLIT_RATIO -1) # 20% split

        # Load annotation file, define split
        if mode == "train":
          start = train_start
          end = train_end
        elif mode  == "val":
          start = val_start
          end = val_end
       
        # Adding to database of base class (Mask RCNN requirement)
        for i in range(start,end+1):
          self.add_image("phone", i, self.img_paths[i])
        print(end-start, " examples")
    
    def load_image(self, image_id):
        # Returns numpy array of image
        return self.images[image_id]

    def image_reference(self, image_id):
        # Not needed
        return ""  
    
    def load_mask(self, image_id):
        '''
        This function returns a list of segmentation masks of the objects
        to be recognised in the image. The Matterpot Mask RCNN takes only
        segmentation masks as input into it. So, we have to convert our
        bounding boxes to segmentation mask. Luckily, skimage provides 
        several functions that make this process easy.
        '''

        masks = []
        class_ids = []
       
        real_x = float(self.X[image_id]) # x coordinate ground truth
        real_y = float(self.Y[image_id]) # y coordinate ground truth

        # Build bounding box
        bbox = np.asarray([round((real_y - self.bboxSize)*self.img_h),
         round((real_x - self.bboxSize)*self.img_w),
          round((real_y +self.bboxSize)*self.img_h),
           round((real_x + self.bboxSize)*self.img_w) ])

        # Build segmentation mask
        start = (bbox[0],bbox[1])
        extent = (bbox[2],bbox[3])
        mask = np.zeros([self.img_h, self.img_w])
        rr, cc = draw.rectangle(start, extent,shape=[self.img_h,self.img_w])
        mask[rr, cc] = 1

        # Return segmentation mask and class id of each mask
        masks.append(mask.astype(np.bool))
        class_ids.append(self.class2index["phone"])
        masks = np.stack(masks, axis=2)
        class_ids = np.asarray(class_ids).astype(dtype=np.int32)

        return masks, class_ids


def parse_dataset(dataset_path):
    '''
    Parses dataset inside dataset_path folder and returns img data
    and x,y coordinates of phone
    Input: dataset_path: Path of dataset
    Returns:
        X: list of x coordinate (float) of phone for each image
        Y: list of y coordiates (float) of phone for each image
        imgs = Numpy array of images
        imgs_path: Meta data, contains file path of each image  
    '''
    filenames = [dataset_path + f"/{constants.LABELS_FILE}"]
    dataset = tf.data.TextLineDataset(filenames)

    # Preparing training data
    imgs = []
    X = []
    Y = []
    Y_regression = []
    with tf.Session() as sess:
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        for i in range(100):
            value = sess.run(next_element)
            vals = value.decode('utf-8').split(" ")
            imgs.append(vals[0])
            X.append(float(vals[1]))
            Y.append(float(vals[2]))

    # getting image data into numpy arrays
    img_paths = imgs.copy()
    for i in range(len(X)): 
        imgs[i] = cv2.imread(dataset_path + "/" + imgs[i])
    imgs = np.stack(imgs,axis=0)

    return X, Y, imgs, img_paths

def get_dataset(file_path):
    X, Y, imgs, img_paths = parse_dataset(file_path)

    config = PhoneConfig()
    
    dataset_train = PhoneDataset()
    dataset_train.load_shapes("train", X, Y, imgs, img_paths)
    dataset_train.prepare()

    dataset_val = PhoneDataset()
    dataset_val.load_shapes("val", X, Y, imgs, img_paths)
    dataset_val.prepare()

    return dataset_train , dataset_val