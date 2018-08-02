import tensorflow as tf
import models
import cv2
import random
import glob, os

fs = glob.glob(os.path.join('data_web', '*.jpg'))
trains = []
for i in range(len(fs)):
    image_name = os.path.join("data_web", "%d.jpg" % (i))
    depth_name = os.path.join("data_web", "%d.png" % (i))
    trains.append((image_name, depth_name))
random.shuffle(trains)

with open('train_web.csv', 'w') as output:
    for (image_name, depth_name) in trains:
        output.write("%s,%s" % (image_name, depth_name))
        output.write("\n")