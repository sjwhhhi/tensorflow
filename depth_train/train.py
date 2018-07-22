import os, glob, cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import models

def load_data(image_dir, image_shape, out_shape):
    fs = glob.glob(os.path.join(image_dir, '*.jpg'))
    x = np.zeros((len(fs),) + image_shape, dtype=np.float32)
    y = np.zeros((len(fs),) + out_shape, dtype=np.float32)
    for i in range(len(fs)):
        img = cv2.imread(fs[i])[:,:,::-1]
        img = img[12:-12,16:-16,:]
        img = cv2.resize(img, (image_shape[1], image_shape[0]))
        img = img.astype('float32')
        x[i] = img
        img = cv2.imread(fs[i][:-3]+'png')
        img = img[12:-12, 16:-16, 0]
        img = cv2.resize(img, (out_shape[1], out_shape[0]))
        img = img.astype('float32')
        y[i,:,:,0] = 0.01*img

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
    return xtrain, xtest, ytrain, ytest

def berHuLoss(y, yp, thr=1):
    loss = tf.where(y>thr, tf.pow(y-yp, 2), tf.abs(y-yp))
    return tf.reduce_mean(loss)

def train(model_data_path, image_dir):
    # Default input size
    height = 240
    width = 320
    channels = 3
    batch_size = 16
    learning_rate = 0.02
    training_epochs = 20
    display_step = 1
    output_height, output_width = height, width
    for i in range(5):
        output_height = np.ceil(output_height / 2)
        output_width = np.ceil(output_width / 2)
    output_height = int(16*output_height)
    output_width = int(16*output_width)

    # Read image
    xtrain, xtest, ytrain, ytest = load_data(image_dir, (height, width, channels),
                                             (output_height, output_width, 1))

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    output_node = tf.placeholder(tf.float32, shape=(None, output_height, output_width, 1))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, True)
    ypred = net.get_output()
    loss = berHuLoss(output_node, ypred)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        # Load the converted parameters
        print('Loading the model')
        saver = tf.train.Saver()
        saver.restore(sess, model_data_path)

        for epoch in range(training_epochs):
            avg_loss = 0
            total_batch = xtrain.shape[0] // batch_size
            for i in range(total_batch):
                _, c = sess.run([optimizer, loss],
                                feed_dict={input_node: xtrain[i * batch_size: (i + 1) * batch_size],
                                           output_node: ytrain[i * batch_size: (i + 1) * batch_size]})
                avg_loss += c
            avg_loss /= total_batch
            plt.plot(epoch + 1, avg_loss, 'co')

            if (epoch + 1) % display_step == 0:
                print("Epoch:", "%04d" % (epoch + 1), "Loss =", avg_loss)
        print("Optimization Finished!")
        saver.save(sess, 'NYU_FCRN_ckpt/model.ckpt')

def main():
    model_path = './NYU_FCRN_ckpt/model.ckpt'
    image_dir = '../data'
    train(model_path, image_dir)

if __name__ == '__main__':
    main()
