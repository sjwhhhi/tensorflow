import os, glob, cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

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
    return x, y

def evaluate(model_data_path, image_path):
    # Default input size
    height = 240
    width = 320
    channels = 3
    output_height, output_width = height, width
    for i in range(5):
        output_height = np.ceil(output_height / 2)
        output_width = np.ceil(output_width / 2)
    output_height = int(16*output_height)
    output_width = int(16*output_width)

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, 1, 1, False)

    with tf.Session() as sess:
        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, model_data_path)

        x, y = load_data(image_path, (height, width, channels), (output_height, output_width, 1))
        ypred = np.zeros(y.shape)

        rel = 0
        total_batch = x.shape[0]
        for i in range(total_batch):
            pred = sess.run(net.get_output(), feed_dict={input_node: x[i: (i + 1)]})
            ypred[i] = pred

    # Evalute the network
    rel = abs(y-ypred) / y
    rel = np.mean(rel)
    thresh = np.maximum((y/ypred), (ypred/y))
    acc = (thresh<1.25).mean()

    return rel, acc

def main():
    model_path = './NYU_FCRN_ckpt/model.ckpt'
    image_path = '../data'
    rel, acc = evaluate(model_path, image_path)
    print('rel: {0}, acc: {1}'.format(rel, acc))


if __name__ == '__main__':
    main()