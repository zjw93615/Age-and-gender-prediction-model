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
    self.files = self.files[:5000]
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
    img = img.resize((32, 32),Image.ANTIALIAS)
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

x = tf.placeholder(tf.float32,shape=[None,32,32,3])
y_true = tf.placeholder(tf.float32,shape=[None,2])

hold_prob = tf.placeholder(tf.float32)

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

convo_1 = convolutional_layer(x,shape=[4,4,3,32])
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling,shape=[4,4,32,64])
convo_2_pooling = max_pool_2by2(convo_2)

convo_2_flat = tf.reshape(convo_2_pooling,[-1,8*8*64])

full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))

full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout,2)

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