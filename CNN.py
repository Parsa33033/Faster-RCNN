
import keras
import numpy as np

class VGG16():
    def __init__(self):
        self.vgg16 = keras.applications.vgg16.VGG16(include_top=False,
                                               weights='imagenet',
                                               input_tensor=None,
                                               input_shape=None,
                                               pooling="max")
    def get_feature_map(self, image, feature_map_layer):
        def get_feature_map_shape(feature_map):
            return feature_map.shape

        # use the layers from input all the way to layer 17 of VGG16
        vgg16_croped = keras.backend.function(inputs=[self.vgg16.layers[0].input],outputs=[self.vgg16.layers[feature_map_layer].output])
        feature_map = np.array(vgg16_croped([image]))
        feature_map_shape = get_feature_map_shape(feature_map)
        return feature_map, feature_map_shape

    def get_num_of_featuremap_channels(self, layer):
        return self.vgg16.layers[layer].output.shape[-1].value
