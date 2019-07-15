
from keras.models import Model
from keras.layers import Conv2D, Input
import tensorflow as tf
import keras.backend as K
from keras.losses import logcosh

from Faster_RCNN.Utils import l1smooth


class RPN():
    def create_model(self , k, number_of_channels):
        """
        shape of the feature map as input is not important because at the end classes_output and regressions_output
        are going to have the same height and width as the input and every image can produce different height and
        width for the feature map
        :param k: number of anchor boxes per anchor(stride)
        :return:
        """
        input = Input(shape=(None, None, number_of_channels))

        out = Conv2D(512, (3*3), padding="same", activation="relu", kernel_initializer="normal", name="rpn_conv")(input)

        class_output = Conv2D(2*k, (1,1), activation="sigmoid", kernel_initializer="uniform", name="rpn_class_output")(out)
        regressor_output = Conv2D(4*k, (1,1), activation="linear", kernel_initializer="zero", name="rpn_regressor_output")(out)

        rpn_model = Model(inputs= input, outputs=[class_output, regressor_output])

        rpn_model.compile(optimizer="adam", loss=["categorical_crossentropy", "logcosh" ])

        return rpn_model
