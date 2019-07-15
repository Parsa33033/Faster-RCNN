
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
import tensorflow as tf
import numpy as np

from Faster_RCNN.Utils import iou_calculation, encode_to_deltas
from Faster_RCNN.Utils import l1smooth


class RCNN():
    def __init__(self, number_of_classes, input_shape=(7,7,512)):
        self.number_of_classes = number_of_classes
        self.input_shape = input_shape
    def create_model(self):
        """
        create rcnn model. since the output is way smaller than the way that the original paper uses rcnn. we use
        two dense layers with 1024 neurons instead of 4096 neurons. logcosh is very similar to l1 smooth as the paper
        suggests using
        :return:
        """
        input = Input(shape=self.input_shape)

        out = Flatten()(input)
        out = Dense(1024, activation="relu", name="rcnn_dense1")(out)
        out = Dropout(0.5)(out)
        out = Dense(1024, activation="relu", name="rcnn_dense2")(out)
        out = Dropout(0.5)(out)

        rcnn_classifier = Dense(self.number_of_classes+1, activation='sigmoid', name='rcnn_classifier')(out)
        rcnn_regressor = Dense(4*(self.number_of_classes), activation='linear', name='rcnn_regressor')(out)

        rcnn_model = Model(inputs=input, outputs=[rcnn_classifier, rcnn_regressor])

        rcnn_model.compile(optimizer="adam", loss=["categorical_crossentropy","logcosh"])

        return rcnn_model


    def create_rcnn_target(self, roi_batch, bounding_boxes, image_class_names, classes, iou_threshold=0.5):
        """
        creating the rcnn targets from batch of regions of interest (roi_batch)
        :param roi_batch: batch of 256 regions of interest
        :param bounding_boxes: all the ground truth bounding boxes in a list
        :param image_class_names: the class labels in a list (should be arranged in a way to point to its corresponding
        bounding box
        :param classes: all the classes in the dataset
        :param iou_threshold:
        :return:
        """
        batch_size = roi_batch.shape[0]
        rcnn_classes = np.zeros((batch_size ,self.number_of_classes+1))
        rcnn_regression = np.zeros((batch_size, 4*(self.number_of_classes)))

        for roi_idx in range(batch_size):
            roi = roi_batch[roi_idx]

            for bbox_idx in range(bounding_boxes.shape[0]):
                bounding_box = bounding_boxes[bbox_idx]
                iou = iou_calculation(roi, bounding_box)

                if iou > iou_threshold:
                    class_index = classes[image_class_names[bbox_idx]]
                    rcnn_classes[roi_idx] = tf.keras.utils.to_categorical(class_index, self.number_of_classes+1)
                    dx, dy, dw, dh = encode_to_deltas(roi, bounding_box)
                    rcnn_regression[roi_idx, 4*(class_index-1)] = dx
                    rcnn_regression[roi_idx, 4*(class_index-1)+1] = dy
                    rcnn_regression[roi_idx, 4*(class_index-1)+2] = dw
                    rcnn_regression[roi_idx, 4*(class_index-1)+3] = dh
                else:
                    v = np.zeros((self.number_of_classes+1))
                    v[0] = 1
                    rcnn_classes[0] = v

        return rcnn_classes, rcnn_regression

