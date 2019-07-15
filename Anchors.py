
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from Faster_RCNN.Utils import iou_calculation, draw_box, xyxy_to_xywh, xywh_to_xyxy

class Anchors:
    def __init__(self,image,
                 image_height,
                 image_width,
                 image_bounding_boxes,
                 feature_map,
                 feature_map_height,
                 feature_map_width,
                 feature_map_num_of_channels,
                 downscale,
                 num_of_anchorboxes_per_anchor,
                 anchor_box_scales,
                 anchor_box_ratios,
                 anchor_box_base_size):
        self.image = image
        self.image_height = image_height
        self.image_width = image_width
        self.image_bounding_boxes = image_bounding_boxes
        self.feature_map = feature_map
        self.feature_map_height = feature_map_height
        self.feature_map_width = feature_map_width
        self.feature_map_num_of_channels = feature_map_num_of_channels
        self.downscale = downscale
        self.k = num_of_anchorboxes_per_anchor
        self.anchor_box_scales = anchor_box_scales
        self.anchor_box_ratios = anchor_box_ratios
        self.anchor_box_base_size = anchor_box_base_size

    def get_rpn_targets(self):
        rpn_target_classes = np.zeros((1, self.feature_map_height, self.feature_map_width, 2 * self.k))
        rpn_target_regressors = np.zeros((1, self.feature_map_height, self.feature_map_width, 4 * self.k))



        #create anchors(strides) by the downscale value for each anchor
        self.anchor_centers, anchor_top_lefts = self.create_anchors()
        self.anchor_boxes = self.create_anchorboxes(self.anchor_centers)


        for anchor_y in range(0, self.image_height, self.downscale):
            for anchor_x in range(0,self.image_width, self.downscale):
                for anchorbox_index in range(len(self.anchor_boxes)):
                    ab = self.anchor_boxes[anchorbox_index]
                    anchor_center = [anchor_x+8, anchor_y+8]
                    point = np.concatenate([anchor_center, anchor_center])
                    anchor_box = ab + point
                    if anchor_box[0]<0 or anchor_box[1]<0 or anchor_box[2]>=self.image_width or anchor_box[3]>=self.image_height:
                        continue
                    for bounding_box in self.image_bounding_boxes:

                        iou = iou_calculation(anchor_box,bounding_box)
                        ay = anchor_y //self.downscale
                        ax = anchor_x //self.downscale
                        if iou>0.5:

                            rpn_target_classes[0, ay, ax, anchorbox_index*2] = 1
                            rpn_target_classes[0, ay, ax, anchorbox_index*2 + 1] = 0

                            # find the deltas(differences) between the current bounding box and anchor box
                            dx, dy, dw, dh = self.encode_to_deltas(anchor_box, bounding_box)

                            rpn_target_regressors[0, ay, ax, anchorbox_index*4] = dx
                            rpn_target_regressors[0, ay, ax, anchorbox_index*4+1] = dy
                            rpn_target_regressors[0, ay, ax, anchorbox_index*4+2] = dw
                            rpn_target_regressors[0, ay, ax, anchorbox_index*4+3] = dh


                        elif iou<0.1:
                            rpn_target_classes[0, ay, ax, anchorbox_index*2] = 0
                            rpn_target_classes[0, ay, ax, anchorbox_index*2 + 1] = 1




        return rpn_target_classes, rpn_target_regressors

    def create_anchors(self):
        x = [i for i in range(0,self.image_width,self.downscale)]
        y = [i for i in range(0,self.image_height,self.downscale)]

        x, y = np.meshgrid(x, y)
        x = np.reshape(x, -1)
        y = np.reshape(y, -1)

        # anchors(strides) with the top-left coordinates
        anchor_top_lefts = np.column_stack([x, y])

        # anchors(strides) with the center coordinates
        anchor_centers = np.column_stack([x+8, y+8])

        return anchor_centers, anchor_top_lefts

    def create_anchorboxes(self, anchor_centers):
        scales, ratios = np.meshgrid(self.anchor_box_scales, self.anchor_box_ratios)

        scales = np.reshape(scales, -1)
        ratios = np.reshape(ratios, -1)
        ratios = np.sqrt(ratios)

        heights = scales * ratios * self.anchor_box_base_size
        widths = scales / ratios * self.anchor_box_base_size
        ch = heights / 2
        cw = widths / 2

        anchorboxes = np.column_stack([
            0 - cw, #xmin
            0 - ch, #ymin
            0 + cw, #xmax
            0 + ch  #ymax
        ])
        return anchorboxes

    def encode_to_deltas(self, anchor_box, bounding_box):
        ax, ay, aw, ah = xyxy_to_xywh(anchor_box)
        bx, by, bw, bh = xyxy_to_xywh(bounding_box)

        dx = (bx - ax)/aw
        dy = (by - ay)/ah
        dw = np.log(bw/aw)
        dh = np.log(bh/ah)

        return dx, dy, dw, dh

    def decode_to_bbox(self, anchor_box, deltas):
        ax, ay, aw, ah = xyxy_to_xywh(anchor_box)
        dx, dy, dw, dh = deltas

        x = dx * aw + ax
        y = dy * ah + ay
        w = (np.e ** dw) * aw
        h = (np.e ** dh) * ah

        xmin, ymin, xmax, ymax = xywh_to_xyxy([x, y, w, h])
        return xmin, ymin, xmax, ymax

