# coding=utf-8

import cifar10
import cifar10_input
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
import numpy as np
# import time

work_dir = '/tmp/cifar10_cnn/1'


max_steps = 20000
batch_size = 128
# max_steps = 1a
# batch_size = 1
KEEPB = 0.5
data_dir = '/Users/zhuxinhui/work/MLPro/tf-test/cifar10_data/cifar-10-batches-bin'


def variable_width_weight_loss(shape, stddev, wl, name):
    # var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    init = tf.truncated_normal(shape, stddev=stddev)
    var = tf.get_variable(initializer=init, name=name)
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

# cifar10.maybe_download_and_extract()


images_train, labels_train = cifar10_input.distorted_inputs(
    data_dir=data_dir, batch_size=batch_size)


images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir, batch_size=batch_size)


class CIFAR_CNN(object):

    def __init__(self, reuseFg):
        self.image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
        self.label_holder = tf.placeholder(tf.int32, [batch_size])

        with tf.variable_scope('w_b_variable', reuse=reuseFg):
            self.weight1 = variable_width_weight_loss(
                shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0, name='vwwl1')
            self.kernel1 = tf.nn.conv2d(self.image_holder, self.weight1, [
                1, 1, 1, 1], padding='SAME')
            # self.bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
            self.bias1 = tf.get_variable(
                initializer=tf.constant(0.0, shape=[64]), name='bias1')
            self.conv1 = tf.nn.relu(tf.nn.bias_add(self.kernel1, self.bias1))
            self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='SAME')
            self.norm1 = tf.nn.lrn(self.pool1, 4, bias=1.0,
                                   alpha=0.001 / 9.0, beta=0.75)

            self.weight2 = variable_width_weight_loss(
                shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0, name='vwwl2')
            self.kernel2 = tf.nn.conv2d(self.norm1, self.weight2, [
                                        1, 1, 1, 1], padding='SAME')
            # self.bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
            self.bias2 = tf.get_variable(
                initializer=tf.constant(0.1, shape=[64]), name='bias2')
            self.conv2 = tf.nn.relu(tf.nn.bias_add(self.kernel2, self.bias2))
            self.norm2 = tf.nn.lrn(self.conv2, 4, bias=1.0,
                                   alpha=0.001 / 9.0, beta=0.75)
            self.pool2 = tf.nn.max_pool(self.norm2, ksize=[1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='SAME')

            self.reshape = tf.reshape(self.pool2, [batch_size, -1])
            self.dim = self.reshape.get_shape()[1].value
            self.weight3 = variable_width_weight_loss(
                shape=[self.dim, 384], stddev=0.04, wl=0.004, name='vwwl3')
            # self.bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
            self.bias3 = tf.get_variable(
                initializer=tf.constant(0.1, shape=[384]), name='bias3')
            self.local3 = tf.nn.relu(
                tf.matmul(self.reshape, self.weight3) + self.bias3)
            self.local3 = tf.nn.dropout(self.local3, KEEPB)

            self.weight4 = variable_width_weight_loss(
                shape=[384, 192], stddev=0.04, wl=0.004, name='vwwl4')
            # self.bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
            self.bias4 = tf.get_variable(
                initializer=tf.constant(0.1, shape=[192]), name='bias4')
            self.local4 = tf.nn.relu(
                tf.matmul(self.local3, self.weight4) + self.bias4)
            self.local4 = tf.nn.dropout(self.local4, KEEPB)

            self.weight5 = variable_width_weight_loss(
                shape=[192, 10], stddev=1 / 192.0, wl=0.0, name='vwwl5')
            # self.bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
            self.bias5 = tf.get_variable(
                initializer=tf.constant(0.0, shape=[10]), name='bias5')
            self.logits = tf.add(
                tf.matmul(self.local4, self.weight5), self.bias5)

    def loss(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

model = CIFAR_CNN('')
model.cost = model.loss(model.logits, model.label_holder)
model.train_op = tf.train.AdamOptimizer(1e-3).minimize(model.cost)
model.top_k_op = tf.nn.in_top_k(model.logits, model.label_holder, 1)

saver = tf.train.Saver()
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

# 训练
for step in range(max_steps):
    # start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([model.train_op, model.cost], feed_dict={
                             model.image_holder: image_batch, model.label_holder: label_batch})
    # duration = time.time() - start_time
    if step % 100 == 0:
        # examples_per_sec = batch_size / duration
        # sec_per_batch = float(duration)
        # format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        # print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
        saver.save(sess, './save/stock.ckpt')
        format_str = ('step %d,loss=%.2f')
        print(format_str % (step, loss_value))

# 测试准确率
# num_examples = 10000
# num_examples = 1
# import math
# num_iter = int(math.ceil(num_examples / batch_size))
# print(num_examples)
# true_count = 0
# total_sample_count = num_iter * batch_size
# step = 0
# while step < num_iter:
#     image_batch, label_batch = sess.run([images_test, labels_test])
#     predictions = sess.run([top_k_op], feed_dict={
#         image_holder: image_batch, label_holder: label_batch})
#     true_count += np.sum(predictions)
#     step += 1
# precision = true_count / float(total_sample_count)
# print('precision @ 1 = %.3f' % precision)

# raise SystemExit

batch_size = 1
KEEPB = 1
images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir, batch_size=batch_size)
model = CIFAR_CNN(True)
model.top_k_op = tf.nn.in_top_k(model.logits, model.label_holder, 1)
saver = tf.train.Saver()
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

module_file = tf.train.latest_checkpoint('./save/')
saver.restore(sess, module_file)

image_batch, label_batch = sess.run([images_test, labels_test])
print(image_batch.shape, label_batch.shape)
logits, predictions = sess.run([model.logits, model.top_k_op], feed_dict={
    model.image_holder: image_batch, model.label_holder: label_batch})
print(logits, predictions)


# raise SystemExit
# 导出模型
print('Exporting trained model to', work_dir)
builder = saved_model_builder.SavedModelBuilder(work_dir)

# Build the signature_def_map.

tensor_info_x = utils.build_tensor_info(model.image_holder)
tensor_info_y = utils.build_tensor_info(model.logits)

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

print('Done exporting!')
