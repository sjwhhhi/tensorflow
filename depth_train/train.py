import tensorflow as tf
import models
from data_process import BatchGenerator
from PIL import Image
import numpy as np

CSV_PATH = 'train_web.csv'
RESTORE_MODEL = 'nyu_model/model.ckpt'
SAVE_MODEL = 'train/model.ckpt'
RESULT_DIR = 'result/'
INITIAL_LR = 0.0001
batch_size = 8
MAX_STEP = 500000


def berHuLoss(label, predict, invalid_depth):
    output_size = 128*160
    predict_all = tf.reshape(predict, [-1, output_size])
    depth_all = tf.reshape(label, [-1, output_size])
    pixel_mask = tf.reshape(invalid_depth, [-1, output_size])
    predict_valid = tf.multiply(predict_all, pixel_mask)
    label_valid = tf.multiply(depth_all, pixel_mask)
    abs_error = tf.abs(label_valid - predict_valid)
    c = 0.2 * tf.reduce_max(abs_error)
    berhuloss = tf.where(abs_error <= c,
                         abs_error,
                         (tf.square(abs_error) + tf.square(c))/(2*c))
    loss = tf.reduce_mean(berhuloss)
    tf.summary.scalar('berhu_loss', loss)
    return loss

def build_loss(scale2_op, depths, pixels_mask):
    output_size = 128*160
    predictions_all = tf.reshape(scale2_op, [-1, output_size])
    depths_all = tf.reshape(depths, [-1, output_size])
    pixels_mask = tf.reshape(pixels_mask, [-1, output_size])
    predictions_valid = tf.multiply(predictions_all, pixels_mask)
    target_valid = tf.multiply(depths_all, pixels_mask)
    d = tf.subtract(predictions_valid, target_valid)
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d, 1)
    sum_d = tf.reduce_sum(d, 1)
    sqare_sum_d = tf.square(sum_d)
    cost = tf.reduce_mean((sum_square_d / output_size) - 0.5 * (sqare_sum_d / pow(output_size,2) ))
    cost = tf.sqrt(cost)
    tf.summary.scalar('loss', cost)
    return cost

def learning_rate(global_step):
    lr = tf.train.exponential_decay(INITIAL_LR,
                                    global_step,
                                    2000,
                                    0.9,
                                    True)
    tf.summary.scalar('learning_rate', lr)
    return lr

def train():
    # load data
    batch_generator = BatchGenerator(batch_size=batch_size)
    image, depth, invalid_depth = batch_generator.csv_input(CSV_PATH)

    # network
    net = models.ResNet50UpProj({'data': image}, batch_size, 1, True)
    logits = net.get_output()

    # loss
    #loss = berHuLoss(depth, logits, invalid_depth)
    loss = build_loss(logits, depth, invalid_depth)

    # fine_tuning
    varlist_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    varlist_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    # learning_rate
    global_step = tf.train.get_or_create_global_step()
    lr = learning_rate(global_step)

    # opt
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
        saver.restore(sess, RESTORE_MODEL)

        #train
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        for i in range(MAX_STEP):
            i = sess.run([global_step])[0]
            _, l = sess.run([opt, loss])
            print('step = %d, loss = %f' % (i, l))
            if i % 10 == 0 and i != 0:
                summary = sess.run(merged)
                writer.add_summary(summary, i)
            if i % 200 == 0 and i != 0:
                _, l, output, gt, rgb = sess.run([opt, loss, logits, depth, image])
                saver.save(sess, SAVE_MODEL)

                #rgb
                img = Image.fromarray(np.uint8(rgb[0]))
                img.save(RESULT_DIR+str(i)+'rgb.png')
                #prediction
                dep = output[0].transpose(2, 0, 1)
                if np.max(dep) != 0:
                    ra_depth = (dep / np.max(dep)) * 255.0
                else:
                    ra_depth = dep * 255.0
                depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
                depth_pil.save(RESULT_DIR+str(i)+'pred.png')
                #groundtruth
                ground = gt[0].transpose(2, 0, 1)
                if np.max(ground) != 0:
                    ra_ground = (ground / np.max(ground)) * 255.0
                else:
                    ra_ground = ground * 255.0
                depth_pil = Image.fromarray(np.uint8(ra_ground[0]), mode="L")
                depth_pil.save(RESULT_DIR+str(i)+'gt.png')

        coord.request_stop()
        coord.join(threads)
        sess.close()

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
