import glob
from PIL import Image
import tensorflow as tf
import pandas as pd
import numpy as np
import random

slim = tf.contrib.slim

PATH = '.\\UTKFace\\*.jpg'


class ImageHelper():
  def __init__(self):
    self.files = glob.glob(PATH)
    random.shuffle(self.files)
    self.i = 0

  def train_test_split(self, percentage=0.95, label='gender'):
    length = len(self.files)

    self.label = label

    self.train_size = int(length * percentage)
    self.test_size = int(length - self.train_size)
    self.train_set = self.files[:self.train_size]
    self.test_set = self.files[self.train_size:]


    print('Preprocessing training data')
    count = 0
    self.train_X = list()
    self.train_y = list()
    for file_name in self.train_set:
      x, y = self.read_image(file_name, label)
      self.train_X.append(x)
      self.train_y.append(y)
      count = count + 1
      if count % 1000 == 0:
        print(count)

    print('Preprocessing testing data')
    self.test_x = list()
    self.test_y = list()
    for file_name in self.test_set:
      x, y = self.read_image(file_name, label)
      self.test_x.append(x)
      self.test_y.append(y)

  def next_batch(self, batch_size):
    x = self.train_X[self.i:self.i+batch_size]
    y = self.train_y[self.i:self.i+batch_size]
      
    self.i = (self.i + batch_size) % self.train_size
    return x, y
    


  def read_image(self, file_name, label):
    img = Image.open(file_name)
    img = img.resize((64, 64),Image.ANTIALIAS)
    img = np.asarray(img, dtype=np.float32) / 255
    file_name = file_name.split('\\')[-1]
    file_name = file_name.split('.')[0]
    file_name = file_name.split('_')
    my_dic = dict()
    my_dic['age'] = np.zeros(10)
    my_dic['age'][min(9,int(int(file_name[0]) / 10))] = 1
    my_dic['gender'] = np.zeros(2)
    my_dic['gender'][int(file_name[1])] = 1
    my_dic['race'] = np.zeros(5)
    my_dic['race'][int(file_name[2])] = 1
    return img, my_dic[label]


img_data = ImageHelper()
img_data.train_test_split()








# Gender Predection

x = tf.placeholder(tf.float32,shape=[None,64,64,3])
y_true = tf.placeholder(tf.float32,shape=[None,2])

def darknet53(inputs):
    """
    Builds Darknet-53 model.
    """
    inputs = _conv2d_fixed_padding(inputs, 32, 3)
    inputs = _conv2d_fixed_padding(inputs, 64, 3, strides=2)
    inputs = _darknet53_block(inputs, 32)
    inputs = _conv2d_fixed_padding(inputs, 128, 3, strides=2)

    for i in range(2):
        inputs = _darknet53_block(inputs, 64)

    inputs = _conv2d_fixed_padding(inputs, 256, 3, strides=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 128)

    # route_1 = inputs
    inputs = _conv2d_fixed_padding(inputs, 512, 3, strides=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 256)

    # route_2 = inputs
    inputs = _conv2d_fixed_padding(inputs, 1024, 3, strides=2)

    for i in range(4):
        inputs = _darknet53_block(inputs, 512)

    return inputs


def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs


def _darknet53_block(inputs, filters):
    shortcut = inputs
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)

    inputs = inputs + shortcut
    return inputs


@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('NHWC' or 'NCHW').
      mode: The mode for tf.pad.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs


inputs = darknet53(x)
inputs = slim.conv2d(inputs,512,[3,3])



inputs = tf.reshape(inputs,[-1,2*2*512])

inputs = slim.fully_connected(inputs,1024)

inputs = slim.fully_connected(inputs,512)

hold_prob = tf.placeholder('float')
inputs = tf.nn.dropout(inputs,hold_prob)

y_pred = slim.fully_connected(inputs,2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(init)

  for i in range(3000):
    batch = img_data.next_batch(100)
    sess.run(train, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})
    
    # PRINT OUT A MESSAGE EVERY 100 STEPS
    if i%100 == 0:
        
      print('Currently on step {}'.format(i))
      print('Accuracy is:')
      # Test the Train Model
      matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

      acc = tf.reduce_mean(tf.cast(matches,tf.float32))

      print(sess.run(acc,feed_dict={x:img_data.test_x,y_true:img_data.test_y,hold_prob:1.0}))
      print('\n')

      saver.save(sess, 'my_net/save_LeNet_5_net.ckpt')

  saver.save(sess, 'my_net/save_LeNet_5_net.ckpt')
  