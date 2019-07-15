
from Faster_RCNN.RPN import RPN
from Faster_RCNN.CNN import VGG16
from Faster_RCNN.Utils import *
from Faster_RCNN.Anchors import Anchors
from Faster_RCNN.ROI import ROI
from Faster_RCNN.ROIPooling import ROIPooling
from Faster_RCNN.RCNN import RCNN
from Faster_RCNN.Utils import xywh_to_xyxy



# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


import numpy as np
from keras.layers import Input
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.models import load_model

class Test():
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
        model_folder = "./Trained_Models/"
        featuremap_num_of_channels = self.vgg16.get_num_of_featuremap_channels(self.vgg_layer)
        # self.rpn_model = RPN().create_model(self.k, featuremap_num_of_channels)
        l1smooth_loss = l1smooth()
        self.rpn_model = load_model(model_folder + "rpn_model.h5", custom_objects={"l1smooth":l1smooth_loss})
        number_of_classes = len(self.classes)
        self.rcnn = RCNN(number_of_classes, input_shape=(*self.roipool_size, featuremap_num_of_channels))
        # self.rcnn_model = self.rcnn.create_model()
        self.rcnn_model = load_model(model_folder + "rcnn_model.h5", custom_objects={"l1smooth":l1smooth_loss})

    def update_image(self, image, image_shape, image_bounding_boxes, image_class_names):
        self.image = image
        self.image_bounding_boxes = image_bounding_boxes
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

        # create rpn regressors and classes target (here targets are 3 dimensional arrays because rpn is fully convolutional)
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

        classes, deltas = self.rpn_model.predict(feature_map)

        r = ROI(anchors, classes, deltas)
        roi, roi_probabilities = r.get_proposals()


        roi_batch = r.get_roi_batch(roi, roi_probabilities, batch_size=256)

        # for r in roi_batch:
        #     xmin,ymin,xmax,ymax = r
        #     for bbox in self.image_bounding_boxes:
        #         if iou_calculation(r,bbox)>0.5:
        #             _,ax = plt.subplots(1)
        #             ax.imshow(np.squeeze(self.image))
        #             ax.add_patch(patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,facecolor='none',linewidth=1, edgecolor='r'))
        #             plt.show()

        roi_batch_downscaled = roi_batch / downscale
        print("feature map shape: ", feature_map.shape)
        roipooling = ROIPooling(mode='tf',pool_size=self.roipool_size)
        pooled_roi_batch = roipooling.get_pooled_rois(feature_map, roi_batch_downscaled)
        print("pooled roid shape: ",pooled_roi_batch.shape)

        # rcnn_target = self.rcnn.create_rcnn_target(roi_batch, self.image_bounding_boxes, self.image_class_names, self.classes)
        # rcnn_classes, rcnn_regression = rcnn_target

        rcnn_classifier, rcnn_regressor = self.rcnn_model.predict(pooled_roi_batch)

        counter = 0
        print(len(rcnn_classifier),"<---")
        for i in range(len(rcnn_classifier)):
            print(rcnn_classifier[i].shape)
            class_idx = np.argmax(rcnn_classifier[i], axis=0)
            if class_idx==0:
                continue
            counter += 1
            print("found box: ",counter)
            dx, dy, dw, dh = rcnn_regressor[i, 4*(class_idx-1):4*(class_idx-1)+4]

            xmin, ymin, xmax, ymax = anchors.decode_to_bbox(roi_batch[i], [dx, dy, dw, dh])
            print(xmin, ymin, xmax, ymax)
            _, ax = plt.subplots(1)
            ax.imshow(np.squeeze(self.image))
            ax.add_patch(patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, facecolor='none', edgecolor='r', linewidth=1))
            plt.show()
