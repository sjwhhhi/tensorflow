import tensorflow as tf
import cifar10_input

data_dir = 'data'
learing_rate = 0.1
batch_size = 64
num_example_per_epoch_for_train = 50000
moving_average_decay = 0.9999
num_epoch_per_decay = 350
learing_rate_decay = 0.1

def distorted_inputs():
    images, labels = cifar10_input.distorted_input(data_dir=data_dir, batch_size = batch_size)
    return images, labels

def _get_variable(name, shape, initializer, wd):
    dtype = tf.float32
    if wd>0:
        regularizer = tf.contrib.layers.l2_regularizer(wd)
    else:
        regularizer = None
    return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, regularizer=regularizer)

def conv(x, ksize, filter_num, stride, padding = 'SAME'):
    filter_in = x.get_shape()[-1]
    shape = [ksize, ksize, filter_in, filter_num]
    weight = _get_variable('weights',
                           shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.1),
                           wd=0.00004)
    return tf.nn.conv2d(x, weight, [1, stride, stride, 1], padding = padding)

def maxpool(x, ksize = 2, strides = 2, padding = 'SAME'):
    return tf.nn.max_pool(x, ksize = [1, ksize, ksize, 1], strides = [1, strides, strides, 1], padding = padding)

def avgpool(x, ksize = 2, strides = 2, padding = 'SAME'):
    return tf.nn.avg_pool(x, ksize = [1, ksize, ksize, 1], strides = [1, strides, strides, 1], padding = padding)

def fc(x, fc_out):
    fc_in = x.get_shape()[1]
    shape = [fc_in, fc_out]
    weight = _get_variable('weights',
                           shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.01),
                           wd=0.00004)
    bias = _get_variable('bias',
                         shape= [fc_out],
                         initializer=tf.constant_initializer(0.0),
                         wd = None)
    y = tf.nn.bias_add(tf.matmul(x, weight), bias)
    return y

def bn(x, is_training):
    shape = x.get_shape()
    pop_mean = tf.get_variable('mean', shape, initializer = tf.constant_initializer(0.0), trainable=False)
    pop_var = tf.get_variable('variance', shape, initializer=tf.constant_initializer(1.0), trainable=False)
    offset = tf.get_variable('beta', shape, initializer=tf.constant_initializer(0.0))
    scale = tf.get_variable('scale', shape, initializer=tf.constant_initializer(1.0))
    epsilon = 1e-4
    decay = 0.999

    if is_training:
        batch_mean, batch_var = tf.nn.moments(x, [0])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            output = tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)
    else:
        output = tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)
    return output

def identity_block(input_layer, output_num, is_training = True):
    shortcut = input_layer
    conv1 = conv(input_layer, ksize=3, filter_num=output_num, stride=1, padding='SAME')
    bn1 = bn(conv1, is_training=is_training)
    relu1 = tf.nn.relu(bn1)
    conv2 = conv(relu1, ksize=3, filter_num=output_num, stride=1, padding='SAME')
    bn2 = bn(conv2, is_training=is_training)
    relu2 = tf.nn.relu(bn2)
    conv3 = conv(relu2, ksize=3, filter_num=output_num, stride=1, padding='SAME')
    bn3 = bn(conv3, is_training=is_training)
    output = shortcut + bn3
    output_act = tf.nn.relu(output)
    return output_act


def con_block(input_layer, output_num1, output_num2, output_num3, is_training = True):
    shortcut = input_layer
    conv1 = conv(input_layer, ksize=1, filter_num=output_num1, stride=1, padding='SAME')
    bn1 = bn(conv1, is_training=is_training)
    relu1 = tf.nn.relu(bn1)
    conv2 = conv(relu1, ksize=3, filter_num=output_num2, stride=1, padding='SAME')
    bn2 = bn(conv2, is_training=is_training)
    relu2 = tf.nn.relu(bn2)
    conv3 = conv(relu2, ksize=1, filter_num=output_num3, stride=1, padding='SAME')
    bn3 = bn(conv3, is_training=is_training)
    shortcut_conv = conv(shortcut, ksize=1, filter_num = output_num3, stride=1, padding='SAME')
    shortcut_bn = bn(shortcut_conv, is_training=is_training)
    output = shortcut_bn + bn3
    output_act = tf.nn.relu(output)
    return output_act

def inference(data, is_training = True):
    with tf.variable_scope('conv1'):
        conv_data = conv(data, ksize=7, filter_num=64, stride=1, padding='SAME')
        max1 = maxpool(conv_data, ksize=3, strides=2, padding='SAME')
    with tf.variable_scope('block1'):
        block1 = con_block(max1, 64, 64, 256, is_training=is_training)
        block2 = con_block(block1, 64, 64, 256, is_training=is_training)
        block3 = con_block(block2, 64, 64, 256,is_training=is_training)
    with tf.variable_scope('block2'):
        block4 = con_block(block3, 128, 128, 512, is_training=is_training)
        block5 = con_block(block4, 128, 128, 512, is_training=is_training)
        block6 = con_block(block5, 128, 128, 512, is_training=is_training)
        block7 = con_block(block6, 128, 128, 512, is_training=is_training)
    with tf.variable_scope('block3'):
        block8 = con_block(block7, 256, 256, 1024, is_training=is_training)
        block9 = con_block(block8, 256, 256, 1024, is_training=is_training)
        block10 = con_block(block9, 256, 256, 1024, is_training=is_training)
        block11 = con_block(block10, 256, 256, 1024, is_training=is_training)
        block12 = con_block(block11, 256, 256, 1024, is_training=is_training)
        block13 = con_block(block12, 256, 256, 1024, is_training=is_training)
    with tf.variable_scope('block4'):
        block14 = con_block(block13, 512, 512, 2048, is_training=is_training)
        block15 = con_block(block14, 512, 512, 2048, is_training=is_training)
        block16 = con_block(block15, 512, 512, 2048, is_training=is_training)
    avg = avgpool(block16, ksize=7, strides=2, padding='SAME')
    fc1 = fc(avg, 10)
    softmax = tf.nn.softmax(fc1, name= 'softmax')
    return softmax

def loss(logit, label):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logit, label)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    losses = cross_entropy_mean + regularization_loss
    tf.summary.scalar('losses', losses)
    return losses

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
