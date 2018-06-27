import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist', one_hot = True)

def weight_varibale(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def accuracy(v_x, v_y):
    global prediction
    pred = sess.run(prediction, feed_dict={x_data: v_x, keep_prob:1})
    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(v_y, 1))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))
    result = sess.run(acc, feed_dict={x_data: v_x, y_data: v_y, keep_prob:1})
    return result

x_data = tf.placeholder(tf.float32, shape = [None, 784])
y_data = tf.placeholder(tf.float32, shape = [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_input = tf.reshape(x_data, [-1, 28, 28, 1])

#conv1_layer
w_conv1 = weight_varibale([5, 5, 1, 32])
b_conv1 = bias_variable([32])
conv1 = tf.nn.relu(conv2d(x_input, w_conv1) + b_conv1)
pool1 = maxpool(conv1)    #14*14

#conv2_layer
w_conv2 = weight_varibale([5, 5, 32, 64])
b_conv2 = bias_variable([64])
conv2 = tf.nn.relu(conv2d(pool1, w_conv2) + b_conv2)
pool2 = maxpool(conv2)    #7*7

#fc1_layer
w_fc1 = weight_varibale([7*7*64, 1024])
b_fc1 = bias_variable([1024])
flat_fc1 = tf.reshape(pool2,[-1, 7*7*64])
relu_fc1 = tf.nn.relu(tf.matmul(flat_fc1, w_fc1) + b_fc1)
dropout = tf.nn.dropout(relu_fc1, keep_prob)

#fc2_layer
w_fc2 = weight_varibale([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(dropout, w_fc2) + b_fc2)

#loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_data * tf.log(prediction), reduction_indices=[1]))
#train
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10001):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x_data: batch_x, y_data: batch_y, keep_prob: 0.5})
        if i%50 == 0:
            print('step = %s, loss = %.3f, accuracy = %.3f' %(i,
                                                              sess.run(cross_entropy, feed_dict={x_data: batch_x, y_data: batch_y, keep_prob:0.5}),
                                                              accuracy(mnist.test.images, mnist.test.labels)))

    save_path = saver.save(sess, "model/lenet.ckpt")

print('Model Saved..')