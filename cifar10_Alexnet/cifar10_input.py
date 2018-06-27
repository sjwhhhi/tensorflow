import os
import tensorflow as tf

image_size = 24
num_class = 10
num_example_per_epoch_for_train = 50000
num_example_per_epoch_for_eval = 10000

def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width *result.depth
    record_bytes = label_bytes + image_bytes
    #reader
    reader = tf.FixedLengthRecordReader(record_bytes= record_bytes)
    result.key, value = reader.read(filename_queue)
    #convert string to uint8
    record_bytes = tf.decode_raw(value, tf.uint8)
    #first bytes represent the label, in int32
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    #remaining bytes represent the image, from [depth*height*width] to [depth,height,width]
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]), [result.depth, result.height, result.width])
    #from [depth, height, width] to [height, width, depth]
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result

def _generate_image_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size = batch_size,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 * batch_size,
            min_after_dequeue = min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return images, tf.reshape(label_batch, [batch_size])

def distorted_input(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' %i)
                 for i in range (1, 6)]
    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    height = image_size
    width = image_size
    #preprocess for trainging
    #get [24,24] part in image
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image,max_delta = 63)
    distorted_image = tf.image.random_contrast(distorted_image, lower = 0.2, upper = 1.8)
    float_image = tf.image.per_image_standardization(distorted_image)
    #set shape for dataset
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])
    min_fraction_queue = 0.4
    min_queue_examples = int(num_example_per_epoch_for_train * min_fraction_queue)

    return _generate_image_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle = True)

def inputs(eval_data, data_dir, batch_size):
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' %i)
                     for i in range(1, 6)]
        num_example_per_epoch = num_example_per_epoch_for_train
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_example_per_epoch = num_example_per_epoch_for_eval

    filenames_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(filenames_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    height = image_size
    width = image_size
    reshaped_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)
    float_image = tf.image.per_image_standardization(reshaped_image)
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])
    min_fraction_queue = 0.4
    min_queue_examples = int(num_example_per_epoch_for_train * min_fraction_queue)
    return _generate_image_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle = False)

