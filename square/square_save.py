# coding=utf-8

'''
导出模型，提供serving使用
建造神经网络，拟合一元二次方程
Weights和biases矩阵，输入是行，输出是列，直观理解列代表该层神经元，假设只有一个包含10个神经元的隐藏层，L0(0,1)->L1(1,10)->L2(10,1)
'''

import os
import sys

import tensorflow as tf
import numpy as np

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat


work_dir = '/tmp/square/1'


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1, 1, 100000)[:, np.newaxis]  # newaxis增加一个维度，原来一维数组边二维数组
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
L1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
predition = add_layer(L1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(
    tf.square(ys - predition), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(20000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if not (i % 100):
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        # print(sess.run(xs, feed_dict={xs: x_data, ys: y_data}))
        # print(sess.run(ys, feed_dict={xs: x_data, ys: y_data}))
        # print(sess.run(predition, feed_dict={xs: x_data, ys: y_data}))

# raise SystemExit


print ('Exporting trained model to', work_dir)
builder = saved_model_builder.SavedModelBuilder(work_dir)

# Build the signature_def_map.

tensor_info_x = utils.build_tensor_info(xs)
tensor_info_y = utils.build_tensor_info(predition)

prediction_signature = signature_def_utils.build_signature_def(
    inputs={'req_x': tensor_info_x},
    outputs={'res_y': tensor_info_y},
    method_name=signature_constants.PREDICT_METHOD_NAME)

legacy_init_op = tf.group(
    tf.initialize_all_tables(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
    sess, [tag_constants.SERVING],
    signature_def_map={
        'predict_x':
            prediction_signature
    },
    legacy_init_op=legacy_init_op)

builder.save()

print ('Done exporting!')
