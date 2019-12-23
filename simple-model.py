import tensorflow as tf
from tensorflow import keras

inputs = tf.keras.Input(shape=(10,))

x = tf.keras.layers.Dense(
            256, activation=tf.nn.tanh,
            kernel_initializer=tf.initializers.random_normal(stddev=1.0),
            bias_initializer=tf.initializers.zeros())(inputs)

model = tf.keras.Model(inputs=inputs, outputs=x)

model.save("test-dir/test-model.h5")