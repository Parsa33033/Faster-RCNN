
from Faster_RCNN.Anchors import Anchors
from Faster_RCNN.Utils import iou_calculation

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

class ROI():
    def __init__(self, anchors, classes, deltas):
        self.anchors = anchors
        self.classes = classes
        self.deltas = deltas

    def get_proposals(self):

        proposals = []
        probabilities = []

        for anchor_y in range(0, self.anchors.image_height, self.anchors.downscale):
            for anchor_x in range(0,self.anchors.image_width, self.anchors.downscale):

                for anchorbox_index in range(len(self.anchors.anchor_boxes)):

                    # create the anchor box of anchor in anchor_x and anchor_y coordinates
                    ab = self.anchors.anchor_boxes[anchorbox_index]
                    anchor_center = [anchor_x+8, anchor_y+8]
                    point = np.concatenate([anchor_center, anchor_center])
                    anchor_box = ab + point

                    #if any anchor boxes passes the image bounderies don't count them
                    if anchor_box[0]<0 or anchor_box[1]<0 or anchor_box[2]>=self.anchors.image_width or anchor_box[3]>=self.anchors.image_height:
                        continue

                    # get the coorditates of the anchor(stride) for the feature map
                    ay = anchor_y //self.anchors.downscale
                    ax = anchor_x //self.anchors.downscale


                    xmin, ymin, xmax, ymax = self.anchors.decode_to_bbox(anchor_box,
                                                self.deltas[0, ay, ax, 4 * anchorbox_index: 4 * anchorbox_index + 4])

                    # if x and y pass the borders of the image crop x and y to align the borders
                    xmin = np.maximum(0, xmin)
                    ymin = np.maximum(0, ymin)
                    xmax = np.minimum(self.anchors.image_width, xmax)
                    ymax = np.minimum(self.anchors.image_height, ymax)

                    if ((xmin - xmax)>=0) or ((ymin - ymax)>=0):
                        continue

                    foreground_prob = self.classes[0, ay, ax, 2* anchorbox_index]
                    background_prob = self.classes[0, ay, ax, 2* anchorbox_index+1]

                    # if the foreground probability is heigher than the back ground probability that means the
                    # region is foreground and should have the index of one, otherwise zero.
                    probability, c = (foreground_prob, 1) if foreground_prob>=background_prob else (background_prob, 0)

                    proposals.append([xmin, ymin, xmax, ymax])
                    probabilities.append([probability, c])

        proposals = np.array(proposals)
        probabilities = np.array(probabilities)

        print("started non max suppression")
        # roi, roi_probabilities = self.non_maximum_suppression(proposals, probabilities)
        selected_indices = tf.image.non_max_suppression(proposals, probabilities[:,0], max_output_size=2000)
        idx = tf.Session().run(selected_indices)
        roi = np.take(proposals, idx, axis= 0)

        roi_probabilities = np.take(probabilities, idx, axis=0)

        return roi, roi_probabilities


    def non_maximum_suppression(self, proposals, probabilities):
        """
        for tutorial purposes, but takes a longer time
        :param proposals:
        :param probabilities:
        :return:
        """
        sorted_index = np.flip(np.argsort(probabilities[:,0]), axis=0)
        delete_index = []
        i = 0
        while i < len(sorted_index):
            print(i)
            print("length of sorted: ", len(sorted_index))

            idx = sorted_index[i]
            highest_prop_proposal = proposals[idx]
            if len(proposals) < 2000:
                break
            j = i + 1
            while j < len(sorted_index):

                iou = iou_calculation(highest_prop_proposal, proposals[sorted_index[j]])
                if iou > 0.5:
                    delete_index.append(sorted_index[j])
                    sorted_index = np.delete(sorted_index, j, axis=0)
                j += 1
            i += 1

        proposals = np.delete(proposals, delete_index, axis=0)
        probabilities = np.delete(probabilities, delete_index, axis=0)

        return proposals, probabilities

    def get_roi_batch(self, roi, roi_probabilities,image,bounding_box, batch_size=256, ratio_of_foregrounds=0.5):
        batch = []
        roi = np.array(roi)
        roi_probabilities = np.array(roi_probabilities)

        foregrounds = roi_probabilities.take(np.argwhere(roi_probabilities[:,1]==1), axis=0)
        backgrounds = roi_probabilities.take(np.argwhere(roi_probabilities[:,1]==0), axis=0)
        foregrounds = np.squeeze(foregrounds)
        backgrounds = np.squeeze(backgrounds)

        f = int(batch_size * ratio_of_foregrounds)

        if len(foregrounds) > f:
            # if size of the foregrounds is as large as half of the batch size, half of the batch size will contain
            # foreground and the other half, backgrounds
            x1 = np.array(roi.take((np.flip(np.argsort(foregrounds[:,0]), axis=0)[0:f]), axis=0))
            b = int(batch_size * (1 - ratio_of_foregrounds))
            x2 = np.array(roi.take((np.flip(np.argsort(backgrounds[:,0]), axis=0)[0:b]), axis=0))
            batch.append(np.concatenate([x1,x2]))
        else:
            # if size of the foregrounds is not as large as half of the batch size use as many foregrounds as you can
            # and the rest of the batch gets filled with backgrounds
            x1 = np.array(roi.take(np.argwhere(roi_probabilities[:,1]==1),axis=0))
            x1 = np.reshape(x1, [x1.shape[0], x1.shape[-1]])
            b = batch_size - int(len(foregrounds))
            x2 = np.array(roi.take((np.flip(np.argsort(backgrounds[:,0]), axis=0)), axis=0))[0:b]
            batch.append(np.concatenate([x1, x2]))

        batch = np.squeeze(np.array(batch))
        return batch
