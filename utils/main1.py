import tensorflow as tf
from model import CaptionGenerator

net = CaptionGenerator()

sess = tf.Session()
net.train(sess,)