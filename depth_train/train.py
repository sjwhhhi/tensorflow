import os, glob, cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import models

def load_data(image_dir, image_shape, out_shape):
    fs = glob.glob(os.path.join(image_dir, '*.jpg'))
    x = np.zeros((len(fs),) + image_shape, dtype=np.float32)
    y = np.zeros((len(fs),) + out_shape, dtype=np.float32)
    for i in range(len(fs)):
        img = cv2.imread(fs[i])[:,:,::-1]
        img = img[12:-12,16:-16,:]
        img = cv2.resize(img, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        img = img.astype('float32')
        x[i] = img
        dep = cv2.imread(fs[i][:-3]+'png')
        dep = dep[12:-12, 16:-16, 0]
        dep = cv2.resize(dep, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_NEAREST)
        dep = np.array(dep).astype('float32')
        y[i,:,:,0] = dep / 255.0

    xtrain, ytrain = x, y
    return xtrain, ytrain

def berHuLoss(label, predict):
    #label_mask = tf.to_float(label>0)
    abs_error = tf.abs(label - predict)
    c = 0.2 * tf.reduce_max(abs_error)
    berhuloss = tf.where(abs_error <= c,
                         abs_error,
                         (tf.square(abs_error) + tf.square(c))/(2*c))
    loss = tf.reduce_mean(berhuloss)
    tf.summary.scalar('berhu_loss', loss)
    return loss

def train(model_data_path, image_dir):
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 8
    learning_rate = 0.0001
    training_epoch = 50000
    display_step = 1
    output_height = 128
    output_width = 160


    # Read image
    xtrain, ytrain = load_data(image_dir, (height, width, channels),
                                           (output_height, output_width, 1))

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    output_node = tf.placeholder(tf.float32, shape=(None, output_height, output_width, 1))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, True)
    ypred = net.get_output()
    loss = berHuLoss(output_node, ypred)

    #fine_tuning
    varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    #opt
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    opt = optimizer.minimize(loss, var_list=varlist[-98:])

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logdir')
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # Load the converted parameters
        print('Loading the model')
        saver = tf.train.Saver()
        saver.restore(sess, model_data_path)

        for epoch in range(training_epoch):
            avg_loss = 0
            total_batch = xtrain.shape[0] // batch_size
            for i in range(total_batch):
                _, c = sess.run([opt, loss],
                                feed_dict={input_node: xtrain[i * batch_size: (i + 1) * batch_size],
                                           output_node: ytrain[i * batch_size: (i + 1) * batch_size]})
                avg_loss += c
                print('Loss = %f' %c)
            avg_loss /= total_batch
            plt.plot(epoch + 1, avg_loss, 'co')
            if (epoch+1) % display_step == 0:
                print("Epoch:", "%04d" % (epoch + 1), "Loss =", avg_loss)
                saver.save(sess, 'train/model.ckpt')

def main():
    model_path = 'nyu_model/model.ckpt'
    image_dir = 'data_aist/'
    train(model_path, image_dir)

if __name__ == '__main__':
    main()
