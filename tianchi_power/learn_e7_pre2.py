# coding=utf-8
# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
Run this script on tensorflow r0.10. Errors appear when using lower versions.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import MySQLdb
import copy

conn = MySQLdb.connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='',
    db='tianchi',
)
cur = conn.cursor()

table_data = []

data_x_mean = 0
data_x_std = 0


# 训练数据
sql_str = "select total,t1,t2,c1,c2 from sum_data where id<='610' order by id"
sum_data = cur.execute(sql_str)
sum_data_info = cur.fetchmany(sum_data)
# print(np.shape(sum_data_info))
for i in range(len(sum_data_info)):
    table_data.append([int(sum_data_info[i][0]), int(sum_data_info[i][1]), int(
        sum_data_info[i][3])])
table_data = np.array(table_data)
# print(np.shape(data_x))
data_x_mean = np.mean(table_data, axis=0)
data_x_std = np.std(table_data, axis=0)
# print(data_x_mean, data_x_std)
normalize_data_x = (table_data - data_x_mean) / data_x_std  # 标准化
normalize_data_x = normalize_data_x.tolist()  # 增加维度
# normalize_data = data_x[:, np.newaxis]
#(609, 1)
# print(np.shape(normalize_data_x))
# print(normalize_data_x)
# raise SystemExit

normalize_data_y = []
for i in range(len(normalize_data_x)):
    normalize_data_y.append([normalize_data_x[i][0]])
normalize_data_x = np.array(normalize_data_x)
normalize_data_y = np.array(normalize_data_y)

print(np.shape(normalize_data_x), np.shape(normalize_data_y))
# print(normalize_data[0])
# raise SystemExit


sql_str = "select total,t1,t2,c1,c2 from sum_data where id>'610' and id<='640' order by id"
pre_sum_data = cur.execute(sql_str)
pre_sum_data_info = cur.fetchmany(pre_sum_data)

# DID_START = 640 + 1

BATCH_START = 0
TIME_STEPS = 20
# 29,31
BATCH_SIZE = len(normalize_data_x) / TIME_STEPS
INPUT_SIZE = 3
OUTPUT_SIZE = 1
CELL_SIZE = 16
LR = 0.0095
# 500
TRAIN_TOTAL = 400
KEEPB = 0.7

LAST_BATCH = []
PREDICT_DATA = []
PREDICT_LEN = len(pre_sum_data_info)


# 生成训练集
# train_x和train_y长度一致，train_x末尾少一个，train_y开始少一个
train_x, train_y = [], []  # 训练集
for i in range(len(normalize_data_x) - TIME_STEPS):
    x = normalize_data_x[i:i + TIME_STEPS]
    y = normalize_data_y[i + 1:i + TIME_STEPS + 1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())
# print(train_x[-1])
# print(train_y[-1])
# raise SystemExit


# def get_batch():
#     global BATCH_START, TIME_STEPS
#     # xs shape (50batch, 20steps)
#     xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS *
#                    BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10 * np.pi)
#     seq = np.sin(xs)
#     res = np.cos(xs)
#     BATCH_START += TIME_STEPS
#     # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
#     # plt.show()
#     # returned seq, res and xs: shape (batch, step, input)
#     return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

def get_batch():
    global BATCH_START
    # xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS *
    # BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10 * np.pi)
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS *
                   BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10 * np.pi)
    seq = np.array(train_x[BATCH_START:(BATCH_START + BATCH_SIZE)])
    res = np.array(train_y[BATCH_START:(BATCH_START + BATCH_SIZE)])
    BATCH_START += BATCH_SIZE
    return [seq, res, xs]

# a, b, x = get_batch()
# print(np.shape(a), np.shape(b))
#((589, 20, 1), (589, 20, 1))
print(np.shape(train_x), np.shape(train_y))
# raise SystemExit


class LSTMRNN(object):

    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, reuseFg):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(
                tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(
                tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden', reuse=reuseFg):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell', reuse=reuseFg):
            self.add_cell()
        with tf.variable_scope('out_hidden', reuse=reuseFg):
            self.add_output_layer()
        # self.compute_cost()
        # self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self,):
        # (batch*n_step, in_size)
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size, ])
        # l_in_y = (batch * n_steps, cell_size)
        l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        l_in_y = tf.nn.dropout(l_in_y, KEEPB)
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(
            l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            self.cell_size, forget_bias=1.0, state_is_tuple=True)
        attn_cell = lstm_cell
        if KEEPB < 1:
            attn_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=KEEPB)
        multy_cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell for _ in range(2)], state_is_tuple=True)
        self.cell_init_state = multy_cell.zero_state(
            self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            multy_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(
            self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    # def compute_cost(self):
    #     losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
    #         [tf.reshape(self.pred, [-1], name='reshape_pred')],
    #         [tf.reshape(self.ys, [-1], name='reshape_target')],
    #         [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
    #         average_across_timesteps=True,
    #         softmax_loss_function=self.ms_error,
    #         name='losses'
    #     )
    #     self.cost = tf.div(
    #         tf.reduce_sum(losses, name='losses_sum'),
    #         self.batch_size,
    #         name='average_cost')
    #     tf.summary.scalar('cost', self.cost)

    def compute_cost(self):
        self.cost = tf.reduce_mean(
            tf.square(tf.reshape(self.pred, [-1]) - tf.reshape(self.ys, [-1])))
        tf.summary.scalar('cost', self.cost)

    def ms_error(self, y_target, y_pre):
        return tf.square(tf.subtract(y_target, y_pre))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        w_v = tf.get_variable(shape=shape, initializer=initializer, name=name)
        return w_v

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


model = LSTMRNN(TIME_STEPS, INPUT_SIZE,
                OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, False)
model.compute_cost()

model.train_op = tf.train.AdamOptimizer(LR).minimize(model.cost)
saver = tf.train.Saver()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

    for j in range(TRAIN_TOTAL):
        BATCH_START = 0

        # if j > 30:
        #     LR = 0.0001
        for i in range(len(train_x) / BATCH_SIZE):
            seq, res, xs = get_batch()
            if i == 0:
                feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    # create initial state
                }
            else:
                feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    model.cell_init_state: state    # use last state as the initial state for this run
                }

            # print(sess.run(model.cell_init_state, feed_dict=feed_dict))

            _, cost, state, pred = sess.run(
                [model.train_op, model.cost,
                    model.cell_final_state, model.pred],
                feed_dict=feed_dict)

            # plotting
            # if j > (TRAIN_TOTAL - 10):
            #     plt.plot(xs[0, :], res[0].flatten(), 'r', xs[
            #         0, :], pred.flatten()[:TIME_STEPS], 'b--')
            #     plt.ylim((-1.2, 1.2))
            #     plt.draw()
            #     plt.pause(0.3)

            if i % 20 == 0:
                # result = sess.run(merged, feed_dict)
                # writer.add_summary(result, i)
                saver.save(sess, './learn_e7_pre2/stock.ckpt')
                print(j, round(cost, 4))

KEEPB = 1
BATCH_SIZE = 1
model = LSTMRNN(TIME_STEPS, INPUT_SIZE,
                OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, True)
saver = tf.train.Saver()
with tf.Session() as sess:
    # 参数恢复
    module_file = tf.train.latest_checkpoint('./learn_e7_pre2/')
    saver.restore(sess, module_file)

    # 取训练集最后一行为测试样本。shape=[1,time_step,input_size]
    # prev_seq = train_y[-1]
    last_start = len(normalize_data_x) - TIME_STEPS
    prev_seq = normalize_data_x[last_start:last_start + TIME_STEPS]
    prev_x = [prev_seq]
    # print(prev_x)
    # raise SystemExit
    predict = []
    # 得到之后100个预测结果
    for i in range(PREDICT_LEN):
        next_seq = sess.run(model.pred, feed_dict={model.xs: prev_x})
        # print(np.shape(next_seq)),[time_step,output_size]
        predict.append(next_seq[-1])
        # 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
        # next_data = [next_seq[-1][0],
        # int(pre_sum_data_info[i][0]), int(pre_sum_data_info[i][1])]

        d_t1 = (pre_sum_data_info[i][1] - data_x_mean[1]) / data_x_std[1]
        d_c1 = (pre_sum_data_info[i][3] - data_x_mean[2]) / data_x_std[2]
        # d_c2 = (pre_sum_data_info[i][3] - data_x_mean[3]) / data_x_std[3]
        next_data = [next_seq[-1][0], d_t1, d_c1]

        next_data = np.array(next_data)
        prev_x[0] = np.vstack((prev_x[0][1:], next_data))
        # print(prev_x)
        # raise SystemExit

    flat_predict = np.array(predict).flatten()
    target_y = flat_predict * data_x_std[0] + data_x_mean[0]
    target_y = np.int32(np.round(target_y))
    # print(target_y)

    predict8 = target_y
    print(predict8)

    target8 = np.int32(np.array(pre_sum_data_info)[:, 0])
    # print(target8)

    n_t = target8
    n_p = np.array(predict8)

    n_c = np.fabs(n_t - n_p)
    loss = np.sum(n_c)
    print(loss)

# for i in range(len(predict8)):
#     d_id = DID_START + i
#     sql_str = "update sum_data set j1='%s' where id='%s'" % (predict8[i], d_id)
#     cur.execute(sql_str)

cur.close()
conn.commit()
conn.close()
