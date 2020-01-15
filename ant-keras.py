import tensorflow as tf
import keras
import numpy as np
import os

# x = tf.concat(actions_input, axis=1)
# dense1_w = tf.get_variable("dense1_w", [28, 128])
# dense1_b = tf.get_variable("dense1_b", [128])
# x = tf.matmul(x, dense1_w) + dense1_b
# x = tf.nn.relu(x)
# dense2_w = tf.get_variable("dense2_w", [128, 64])
# dense2_b = tf.get_variable("dense2_b", [64])
# x = tf.matmul(x, dense2_w) + dense2_b
# x = tf.nn.relu(x)
# final_w = tf.get_variable("final_w", [64, 8])
# final_b = tf.get_variable("final_b", [8])
# x = tf.matmul(x, final_w) + final_b
# pi = x
# self.pi = pi

take_weights_here = None

if take_weights_here is None:
    take_weights_here = {}
    exec(open("RoboschoolAnt_v1_2017jul.weights").read(), take_weights_here)

w_1 = [take_weights_here["weights_dense1_w"], take_weights_here["weights_dense1_b"]]
w_2 = [take_weights_here["weights_dense2_w"], take_weights_here["weights_dense2_b"]]
w_f = [take_weights_here["weights_final_w"], take_weights_here["weights_final_b"]]


inputs = keras.Input((28,))
l_dense1 = keras.layers.Dense(128, activation=tf.nn.relu, use_bias=True, kernel_initializer=keras.initializers.random_normal(), bias_initializer=keras.initializers.random_normal())
#l_dense1.set_weights(w_1)
l_dense2 = keras.layers.Dense(64, activation=tf.nn.relu, use_bias=True, kernel_initializer=keras.initializers.random_normal(), bias_initializer=keras.initializers.random_normal())
#l_dense2.set_weights(w_2)
l_dense3 = keras.layers.Dense(8, use_bias=True, kernel_initializer=keras.initializers.random_normal(), bias_initializer=keras.initializers.random_normal())
#l_dense3.set_weights(w_f)


l1 = l_dense1(inputs)
l2 = l_dense2(l1)
l3 = l_dense3(l2)

model = keras.Model(inputs=inputs, outputs=l3)

#test = model.get_layer(index=0)
#model = tf.keras.models.Sequential()
# model.add(l_dense1)
# model.add(l_dense2)
# model.add(l_dense3)
#model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.mean_absolute_error)
#model.build(input_shape=(None, 28))
model.get_layer(index=1).set_weights(w_1)
model.get_layer(index=2).set_weights(w_2)
model.get_layer(index=3).set_weights(w_f)
model.compile(keras.optimizers.Adam(), loss=keras.losses.mean_squared_error)

model.save("test-dir/agentzoo-conversion/keras-ant-no_tf.h5")