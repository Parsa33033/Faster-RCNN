


from Faster_RCNN.Start import Start
from Faster_RCNN.Train import Train
from Faster_RCNN.Test import Test

import numpy as np


# training or testing phase
phase = "training"
# phase = "testing"

training_folder = "./Stanford_Dataset/cars_train/"
testing_folder = "./Stanford_Dataset/cars_test/"

# get images from dataset
anchorbox_scales = [1, 2, 3]
anchorbox_ratios = [0.5, 1, 2]
anchorbox_base_size = 128


number_of_anchorboxes_per_anchor = len(anchorbox_ratios)*len(anchorbox_scales)

start = Start()


if phase=='training':
    classes = start.get_classes()
    training = Train(number_of_anchorboxes_per_anchor,
                 anchorbox_scales,
                 anchorbox_ratios,
                 anchorbox_base_size,
                 classes)

    start.get_instace(training)
    start.start_training_dataset(training_folder)
elif phase=='testing':
    classes = start.get_classes()
    testing = Test(number_of_anchorboxes_per_anchor,
                 anchorbox_scales,
                 anchorbox_ratios,
                 anchorbox_base_size,
                 classes)

    start.get_instace(testing)
    start.start_testing_dataset(testing_folder)




