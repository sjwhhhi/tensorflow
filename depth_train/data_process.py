import tensorflow as tf

INPUT_HEIGHT = 228
INPUT_WIDTH = 304
OUTPUT_HEIGHT = 128
OUTPUT_WIDTH = 160

class BatchGenerator:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def csv_input(self, csv_file):
        filename_queue = tf.train.string_input_producer([csv_file], shuffle=True)
        reader = tf.TextLineReader()
        _, data_example = reader.read(filename_queue)
        image_examples, depth_targets = tf.decode_csv(data_example, [["path"], ["annotation"]])
        #input
        jpg = tf.read_file(image_examples)
        image = tf.image.decode_jpeg(jpg, channels=3)
        image = tf.cast(image, tf.float32)
        #target
        png = tf.read_file(depth_targets)
        depth = tf.image.decode_png(png, channels=1)
        depth = tf.cast(depth, tf.float32)
        depth = tf.div(depth, [255.0])
        #resize
        image = tf.image.resize_images(image, (INPUT_HEIGHT, INPUT_WIDTH))
        depth = tf.image.resize_images(depth, (OUTPUT_HEIGHT, OUTPUT_WIDTH))
        #invalid depth
        invalid_depth = tf.sign(depth)
        images, depths, invalid_depths = tf.train.batch(
            [image, depth, invalid_depth],
            batch_size=self.batch_size,
            num_threads=4,
            capacity=50 + 3 * self.batch_size,
        )
        return images, depths, invalid_depths