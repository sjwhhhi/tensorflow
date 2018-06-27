import tensorflow as tf
import cifar10
import os

train_dir = 'train'
max_step = 100000

def train():
    with tf.Graph().as_default():
      global_step = tf.train.get_or_create_global_step()
      images, labels = cifar10.distorted_inputs()
      logits = cifar10.inference(images, train = True)
      loss = cifar10.loss(logits, labels)
      accuracy = cifar10.accuracy(logits, labels)
      train_op = cifar10.train(loss, global_step)

      class _LoggerHook(tf.train.SessionRunHook):

        def begin(self):
          self._step = -1

        def before_run(self, run_context):
          self._step += 1
          return tf.train.SessionRunArgs([loss, accuracy])

        def after_run(self, run_context, run_values):
          if self._step % 10 == 0:
            loss_value, acc_value = run_values.results
            format_str = ('step %d, loss = %.2f, accuracy = %.2f ')
            print (format_str %(self._step, loss_value, acc_value))

      with tf.train.MonitoredTrainingSession(
          checkpoint_dir=train_dir,
          hooks=[tf.train.StopAtStepHook(last_step=max_step),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
          config=tf.ConfigProto(
            log_device_placement=False)) as mon_sess:
          while not mon_sess.should_stop():
            mon_sess.run(train_op)

def main(argv=None):
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
