# -*- coding: utf-8 -*-
"""Keras model parser.

@author: rbodo
"""

import numpy as np
#import keras.backend as k
#import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as k
from snntoolbox.parsing.utils import AbstractModelParser


class ModelParser(AbstractModelParser):

    def get_layer_iterable(self):
        return self.input_model.layers

    def get_type(self, layer):
        from snntoolbox.parsing.utils import get_type
        return get_type(layer)

    def get_batchnorm_parameters(self, layer):
        mean = k.get_value(layer.moving_mean)
        var = k.get_value(layer.moving_variance)
        var_eps_sqrt_inv = 1 / np.sqrt(var + layer.epsilon)
        gamma = np.ones_like(mean) if layer.gamma is None else \
            k.get_value(layer.gamma)
        beta = np.zeros_like(mean) if layer.beta is None else \
            k.get_value(layer.beta)
        axis = layer.axis

        return [mean, var_eps_sqrt_inv, gamma, beta, axis]

    def get_inbound_layers(self, layer):
        from snntoolbox.parsing.utils import get_inbound_layers
        return get_inbound_layers(layer)

    @property
    def layers_to_skip(self):
        # noinspection PyArgumentList
        return AbstractModelParser.layers_to_skip.fget(self)

    def has_weights(self, layer):
        from snntoolbox.parsing.utils import has_weights
        return has_weights(layer)

    def initialize_attributes(self, layer=None):
        attributes = AbstractModelParser.initialize_attributes(self)
        attributes.update(layer.get_config())
        return attributes

    def get_input_shape(self):
        return tuple(self.get_layer_iterable()[0].input_shape[0][1:])

        #return tuple(self.get_layer_iterable()[0].batch_input_shape[1:])

    def get_output_shape(self, layer):
        return layer.output_shape

    def parse_dense(self, layer, attributes):
        attributes['parameters'] = layer.get_weights()
        if layer.bias is None:
            attributes['parameters'].append(np.zeros(layer.output_shape[1]))
            attributes['use_bias'] = True

    def parse_convolution(self, layer, attributes):
        attributes['parameters'] = layer.get_weights()
        if layer.bias is None:
            attributes['parameters'].append(np.zeros(layer.filters))
            attributes['use_bias'] = True
        assert layer.data_format == k.image_data_format(), (
            "THe input model was setup with image data format '{}', but your "
            "keras config file expects '{}'.".format(layer.data_format,
                                                     k.image_data_format()))

    def parse_depthwiseconvolution(self, layer, attributes):
        attributes['parameters'] = layer.get_weights()
        if layer.bias is None:
            a = 1 if layer.data_format == 'channels_first' else -1
            attributes['parameters'].append(np.zeros(layer.depth_multiplier *
                                                     layer.input_shape[a]))
            attributes['use_bias'] = True

    def parse_pooling(self, layer, attributes):
        pass

    def get_activation(self, layer):

        return layer.activation.__name__

    def get_outbound_layers(self, layer):

        from snntoolbox.parsing.utils import get_outbound_layers

        return get_outbound_layers(layer)

    def parse_concatenate(self, layer, attributes):
        pass


def load(path, filename, **kwargs):
    """Load network from file.

    Parameters
    ----------

    path: str
        Path to directory where to load model from.

    filename: str
        Name of file to load model from.

    Returns
    -------

    : dict[str, Union[keras.models.Sequential, function]]
        A dictionary of objects that constitute the input model. It must
        contain the following two keys:

        - 'model': keras.models.Sequential
            Keras model instance of the network.
        - 'val_fn': function
            Function that allows evaluating the original model.
    """

    import os
    import tensorflow as tf
    from tensorflow.keras import models, metrics
    #from keras import models, metrics

    filepath = str(os.path.join(path, filename))

    if os.path.exists(filepath + '.json'):
        model = models.model_from_json(open(filepath + '.json').read())
        model.load_weights(filepath + '.h5')
        # With this loading method, optimizer and loss cannot be recovered.
        # Could be specified by user, but since they are not really needed
        # at inference time, set them to the most common choice.
        # TODO: Proper reinstantiation should be doable since Keras2
        model.compile('sgd', 'categorical_crossentropy',
                      ['accuracy', metrics.top_k_categorical_accuracy])
    else:
        from snntoolbox.parsing.utils import get_custom_activations_dict
        filepath_custom_objects = kwargs.get('filepath_custom_objects', None)
        if filepath_custom_objects is not None:
            filepath_custom_objects = str(filepath_custom_objects)  # python 2

        #########################################################################
        class Normc_initializer(tf.keras.initializers.Initializer):
            def __init__(self, std=1.0):
                self.std = std

            def __call__(self, shape, dtype=None, partition_info=None):
                out = np.random.randn(*shape).astype(np.float32)
                out *= self.std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
                return tf.constant(out)

        class ObservationNormalizationLayer(tf.keras.layers.Layer):
            def __init__(self, ob_mean, ob_std, **kwargs):
                self.ob_mean = ob_mean
                self.ob_std = ob_std
                super(ObservationNormalizationLayer, self).__init__(**kwargs)

            def call(self, x):
                return tf.clip_by_value((x - self.ob_mean) / self.ob_std, -5.0, 5.0)

            # get_config and from_config need to implemented to be able to serialize the model
            def get_config(self):
                base_config = super(ObservationNormalizationLayer, self).get_config()
                base_config['ob_mean'] = self.ob_mean
                base_config['ob_std'] = self.ob_std
                return base_config

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        class DiscretizeActionsUniformLayer(tf.keras.layers.Layer):
            def __init__(self, num_ac_bins, adim, ahigh, alow, **kwargs):
                self.num_ac_bins = num_ac_bins
                self.adim = adim
                # ahigh, alow are NumPy arrays when extracting from the environment, but when the model is loaded from a h5
                # File they get initialised as a normal list, where operations like subtraction does not work, thereforce
                # cast them explicitly
                self.ahigh = np.array(ahigh)
                self.alow = np.array(alow)
                super(DiscretizeActionsUniformLayer, self).__init__(**kwargs)

            def call(self, x):
                # Reshape to [n x i x j] where n is dynamically chosen, i equals action dimension and j equals the number
                # of bins
                scores_nab = tf.reshape(x, [-1, self.adim, self.num_ac_bins])
                # This picks the bin with the greatest value
                a = tf.argmax(scores_nab, 2)

                # Then transform the interval from [0, num_ac_bins - 1] to [-1, 1] which equals alow and ahigh
                ac_range_1a = (self.ahigh - self.alow)[None, :]
                return 1. / (self.num_ac_bins - 1.) * tf.keras.backend.cast(a, 'float32') * ac_range_1a + self.alow[None, :]

                # get_config and from_config need to implemented to be able to serialize the model

            def get_config(self):
                base_config = super(DiscretizeActionsUniformLayer, self).get_config()
                base_config['num_ac_bins'] = self.num_ac_bins
                base_config['adim'] = self.adim
                base_config['ahigh'] = self.ahigh
                base_config['alow'] = self.alow
                return base_config

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        custom_objects = {'Normc_initializer': Normc_initializer,
                          'ObservationNormalizationLayer': ObservationNormalizationLayer,
                          'DiscretizeActionsUniformLayer': DiscretizeActionsUniformLayer}

        custom_activation_dict = get_custom_activations_dict(filepath_custom_objects)
        model = models.load_model(
            str(filepath + '.h5'),
            #get_custom_activations_dict(filepath_custom_objects))
            custom_objects=custom_objects)
        model.compile(tf.keras.optimizers.Adam(), tf.keras.losses.MeanSquaredError()) #model.optimizer, model.loss,
                      #['accuracy', metrics.top_k_categorical_accuracy])

    return {'model': model, 'val_fn': model.evaluate}


def evaluate(val_fn, batch_size, num_to_test, x_test=None, y_test=None,
             dataflow=None):
    """Evaluate the original ANN.

    Can use either numpy arrays ``x_test, y_test`` containing the test samples,
    or generate them with a dataflow
    (``Keras.ImageDataGenerator.flow_from_directory`` object).

    Parameters
    ----------

    val_fn:
        Function to evaluate model.

    batch_size: int
        Batch size

    num_to_test: int
        Number of samples to test

    x_test: Optional[np.ndarray]

    y_test: Optional[np.ndarray]

    dataflow: keras.ImageDataGenerator.flow_from_directory
    """

    if x_test is not None:
        score = val_fn(x_test, y_test, batch_size, verbose=0)
    else:
        score = np.zeros(3)
        batches = int(num_to_test / batch_size)
        for i in range(batches):
            x_batch, y_batch = dataflow.next()
            score += val_fn(x_batch, y_batch, batch_size, verbose=0)
        score /= batches

    #print("Top-1 accuracy: {:.2%}".format(score[1]))
    #print("Top-5 accuracy: {:.2%}\n".format(score[2]))
    print("Score: {}".format(score))
    #return score[1]
    return 0.0