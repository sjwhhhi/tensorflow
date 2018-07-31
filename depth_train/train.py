import os, glob, cv2
import numpy as np
import tensorflow as tf
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
    #label_mask = tf.to_float(label != 0)
    abs_error = tf.abs(label - predict)
    c = 0.2 * tf.reduce_max(abs_error)
    berhuloss = tf.where(abs_error <= c,
                         abs_error,
                         (tf.square(abs_error) + tf.square(c))/(2*c))
    loss = tf.reduce_mean(berhuloss)
    tf.summary.scalar('berhu_loss', loss)
    return loss

def learning_rate(init_lr, step):
    learning_rate = tf.train.exponential_decay(init_lr,
                                               global_step=step,
                                               decay_steps=2000,
                                               decay_rate=0.8,
                                               staircase=True)
    tf.summary.scalar('lr', learning_rate)
    return learning_rate

def train(model_data_path, image_dir):
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 8
    init_learning_rate = 0.0001
    training_epoch = 500000
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
    varlist_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    varlist_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    #learning_rate
    global_step = tf.train.get_or_create_global_step()
    lr = learning_rate(init_learning_rate, global_step)

    #opt
    optimizer = tf.train.AdamOptimizer(lr)
    opt = optimizer.minimize(loss, global_step=global_step, var_list=varlist_train[-98:])

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logdir')
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # Load the converted parameters
        print('Loading the model')
        saver = tf.train.Saver(var_list=varlist_all)
        saver.restore(sess, model_data_path)

        for step in range(training_epoch):
            offset = (step * batch_size) % (xtrain.shape[0] - batch_size)
            _, c = sess.run([opt, loss],
                            feed_dict={input_node: xtrain[offset:(offset + batch_size)],
                                       output_node: ytrain[offset:(offset + batch_size)]})
            print('step = %d, loss = %f' % (step, c))
            if step % 10 == 0:
                _, c, summary = sess.run([opt, loss, merged],
                                feed_dict={input_node: xtrain[offset:(offset + batch_size)],
                                           output_node: ytrain[offset:(offset + batch_size)]})
                writer.add_summary(summary, step)
            if step % 200 == 0 and step != 0:
                saver.save(sess, 'train/model.ckpt')

def main():
    model_path = 'nyu_model/model.ckpt'
    image_dir = 'data_mix/'
    train(model_path, image_dir)

if __name__ == '__main__':
    main()
