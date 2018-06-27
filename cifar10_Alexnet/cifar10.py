import tensorflow as tf
import cifar10_input

data_dir = 'data'
learing_rate = 0.1
batch_size = 128
num_example_per_epoch_for_train = 50000
moving_average_decay = 0.9999
num_epoch_per_decay = 350
learing_rate_decay = 0.1

def distorted_inputs():
    images, labels = cifar10_input.distorted_input(data_dir=data_dir, batch_size = batch_size)
    return images, labels

def _variable_on_gpu(name, shape, initializer):
  with tf.device('/gpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  dtype = tf.float32
  var = _variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inference(data, train = False):
    #conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape = [5,5,3,96], stddev = 5e-2, wd = None)
        conv = tf.nn.conv2d(data, kernel, [1,4,4,1], padding='SAME')
        bias = _variable_on_gpu('bias', [96], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, bias)
        conv1 = tf.nn.relu(pre_activation, name =scope.name)
    pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name = 'pool1')

    #conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5,5,96,256], stddev= 5e-2, wd=None)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        bias = _variable_on_gpu('bias', [256], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, bias)
        conv2 = tf.nn.relu(pre_activation, name = scope.name)
    pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool2')

    #conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights', shape = [3,3,256,384], stddev = 5e-2, wd=None)
        conv = tf.nn.conv2d(pool2, kernel, [1,1,1,1], padding='SAME')
        bias = _variable_on_gpu('bias', [384], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, bias)
        conv3 = tf.nn.relu(pre_activation, name = scope.name)
    pool3 = tf.nn.max_pool(conv3, ksize=[1,3,3,1], strides=[1,2,2,1], padding= 'SAME', name='pool3')

    #conv4
    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights', shape = [3,3,384,384], stddev = 5e-2, wd=None)
        conv = tf.nn.conv2d(pool3, kernel, [1,1,1,1], padding='SAME')
        bias = _variable_on_gpu('bias', [384], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, bias)
        conv4 = tf.nn.relu(pre_activation, name = scope.name)
    pool4 = tf.nn.max_pool(conv4, ksize=[1,3,3,1], strides=[1,2,2,1], padding= 'SAME', name='pool4')

    #conv5
    with tf.variable_scope('conv5') as scope:
        kernel = _variable_with_weight_decay('weights', shape = [3,3,384,256], stddev = 5e-2, wd=None)
        conv = tf.nn.conv2d(pool4, kernel, [1,1,1,1], padding='SAME')
        bias = _variable_on_gpu('bias', [256], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, bias)
        conv5 = tf.nn.relu(pre_activation, name = scope.name)
    pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding= 'SAME', name='pool5')

    #fc
    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(pool5, [data.get_shape().as_list()[0], -1])
        dim = reshape.get_shape()[1].value
        kernel = _variable_with_weight_decay('weights', shape = [dim, 384],
                                             stddev=0.04, wd=0.004)
        bias = _variable_on_gpu('bias', [384], tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, kernel)+bias, name=scope.name)
        if train == True:
            drop1 = tf.nn.dropout(fc1, keep_prob=0.5, name='dropout')
        else:
            drop1 = tf.nn.dropout(fc1, keep_prob = 1, name='dropout')
    #fc2
    with tf.variable_scope('fc2') as scope:
        kernel = _variable_with_weight_decay('weights', shape = [384, 192],
                                            stddev=0.04, wd=0.004)
        bias = _variable_on_gpu('bias', [192], tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(drop1, kernel) + bias, name=scope.name)

    #softmax
    with tf.variable_scope('softmax_linear') as scope:
        kernel = _variable_with_weight_decay('weights', [192, 10],
                                             stddev=0.04, wd= None)
        bias = _variable_on_gpu('bias', [10], tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc2, kernel), bias, name=scope.name)
        softmax = tf.nn.softmax(softmax_linear, name='softmax')
    return softmax

def loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  tf.summary.scalar('losses', cross_entropy_mean)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def accuracy(logits, labels):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', acc)
    return acc

def train(loss, global_step):
    num_batch = num_example_per_epoch_for_train / batch_size
    decay_step = int(num_batch*num_epoch_per_decay)
    lr = tf.train.exponential_decay(learing_rate,
                                    global_step,
                                    decay_step,
                                    learing_rate_decay,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss=loss, global_step=global_step)
    return train_step
