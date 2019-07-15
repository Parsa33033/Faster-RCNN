
from Faster_RCNN.RPN import RPN
from Faster_RCNN.CNN import VGG16
from Faster_RCNN.Utils import *
from Faster_RCNN.Anchors import Anchors
from Faster_RCNN.ROI import ROI
from Faster_RCNN.ROIPooling import ROIPooling
from Faster_RCNN.RCNN import RCNN

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


import numpy as np
from keras.layers import Input
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.models import load_model

class Train():
    def __init__(self, number_of_anchorboxes_per_anchor, anchor_box_scales, anchor_box_ratios, anchor_box_base_size, classes, roipool_size=(7,7), epochs=1):
        self.k = number_of_anchorboxes_per_anchor
        self.anchor_box_scales = anchor_box_scales
        self.anchor_box_ratios = anchor_box_ratios
        self.anchor_box_base_size = anchor_box_base_size
        self.classes = classes
        self.roipool_size = roipool_size
        self.epochs = epochs
        self.vgg16 = VGG16()
        self.vgg_layer = 17
        self.rpn_model_dir = "./Trained_Models/rpn_model.h5"
        self.rcnn_model_dir = "./Trained_Models/rcnn_model.h5"
        featuremap_num_of_channels = self.vgg16.get_num_of_featuremap_channels(self.vgg_layer)

        # if a rpn model already exists in Trained_Models folder load the model, otherwise create a new model
        try:
            self.rpn_model = load_model(self.rpn_model_dir)
            print("existing rpn model loaded!")
        except:
            self.rpn_model = RPN().create_model(self.k, featuremap_num_of_channels)
        number_of_classes = len(self.classes)
        self.rcnn = RCNN(number_of_classes, input_shape=(*self.roipool_size, featuremap_num_of_channels))

        # if a rcnn model already exists in Trained_Models folder load the model, otherwise create a new model
        try:
            self.rcnn_model = load_model(self.rcnn_model_dir)
            print("existing rcnn model loaded!")
        except:
            self.rcnn_model = self.rcnn.create_model()

    def update_image(self, image, image_shape, image_bounding_boxes, image_class_names):
        self.image = image
        self.image_bounding_boxes = image_bounding_boxes

        # prepare image such as shape (smaller side becomes 600 pixels)
        self.image, self.image_height, self.image_width, self.image_bounding_boxes = prepare_image_and_bboxes(self.image,
                                                                                                      self.image_bounding_boxes)
        self.image_class_names = image_class_names
        self.train()

    def train(self):

        # self.show_image_with_bbox()

        # get the feature map of 17th layer of vgg16 from the image
        feature_map, feature_map_shape = self.vgg16.get_feature_map(self.image, self.vgg_layer)
        feature_map = reshape_for_CNN(np.squeeze(feature_map))
        feature_map_num_of_channels = feature_map_shape[-1]
        feature_map_height = feature_map_shape[-3]
        feature_map_width = feature_map_shape[-2]

        assert self.image_height//feature_map_height == self.image_width//feature_map_width
        downscale = self.image_height//feature_map_height

        ## start train the RPN by the feature map and anchor box regressors and classes(foreground or background classes)

        # create rpn regressors and classes as the target for training(here targets are 3 dimensional arrays because rpn is fully convolutional)
        anchors = Anchors(self.image,
                          self.image_height,
                          self.image_width,
                          self.image_bounding_boxes,
                          feature_map,
                          feature_map_height,
                          feature_map_width,
                          feature_map_num_of_channels,
                          downscale,
                          self.k,
                          self.anchor_box_scales,
                          self.anchor_box_ratios,
                          self.anchor_box_base_size)

        rpn_target_classes, rpn_target_regressor = anchors.get_rpn_targets()

        # training rpn model
        self.rpn_model.fit(feature_map, [rpn_target_classes, rpn_target_regressor], epochs=self.epochs)
        print("saving rpn model...")
        self.rpn_model.save(self.rpn_model_dir)
        print("rpn model saved!")

        # predict regions of interest (roi)
        classes, deltas = self.rpn_model.predict(feature_map)

        r = ROI(anchors, classes, deltas)
        roi, roi_probabilities = r.get_proposals()

        roi_batch = r.get_roi_batch(roi, roi_probabilities,self.image,self.image_bounding_boxes)

        roi_batch_downscaled = roi_batch / downscale
        print("feature map shape: ", feature_map.shape)

        #roi pooling
        roipooling = ROIPooling(mode='tf',pool_size=self.roipool_size)
        pooled_roi_batch = roipooling.get_pooled_rois(feature_map, roi_batch_downscaled)

        print("pooled roid shape: ",pooled_roi_batch.shape)


        ## start training rcnn

        rcnn_target = self.rcnn.create_rcnn_target(roi_batch, self.image_bounding_boxes, self.image_class_names, self.classes)
        rcnn_classes, rcnn_regression = rcnn_target
        self.rcnn_model.fit(pooled_roi_batch, [rcnn_classes, rcnn_regression], epochs=self.epochs)
        print("saving rcnn model...")
        self.rcnn_model.save(self.rcnn_model_dir)
        print("rcnn model saved!")
