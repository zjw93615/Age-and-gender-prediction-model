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

  def train_test_split(self, percentage=0.995, label='gender'):
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
    img = img.resize((224, 224),Image.ANTIALIAS)
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







#VGG16 model

def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_16.default_image_size = 224


x = tf.placeholder(tf.float32,shape=[None,224,224,3])
y_true = tf.placeholder(tf.float32,shape=[None,2])

hold_prob = tf.placeholder(tf.float32)

y_pred, _ = vgg_16(x, num_classes=2, dropout_keep_prob=hold_prob)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

saver = tf.train.Saver()



gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tf.ConfigProto(gpu_options=gpu_options)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
  sess.run(init)

  for i in range(3000):
    batch = img_data.next_batch(30)
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

      saver.save(sess, 'my_net/save_VGG_16_net.ckpt')

  saver.save(sess, 'my_net/save_VGG_16_net.ckpt')