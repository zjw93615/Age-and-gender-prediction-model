import tensorflow as tf
import numpy as np
import argparse
import cv2

import glob
import random

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", default='deploy.prototxt.txt',
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default='res10_300x300_ssd_iter_140000.caffemodel',
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()



def prediction(images):
  # x = tf.placeholder(tf.float32,shape=[None,64,64,3])

  # hold_prob = tf.placeholder(tf.float32)
  # saver = tf.train.Saver()
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
  config = tf.ConfigProto(gpu_options=gpu_options)
  # config = tf.ConfigProto()
  # config.gpu_options.allow_growth = True
  with tf.Session(config = config) as sess:
    saver = tf.train.import_meta_graph('my_net/save_my_net.ckpt.meta')
    saver.restore(sess, 'my_net/save_my_net.ckpt')


    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    hold_prob = graph.get_tensor_by_name("hold_prob:0")
    y_pred = graph.get_tensor_by_name('y_prediction:0')

    # Prediction
    pred = tf.argmax(y_pred,1)

    result = sess.run(pred,feed_dict={x: images,hold_prob:1.0})

  return result



faces = list()
location = list()
# loop over the detections
for i in range(0, detections.shape[2]):
  # extract the confidence (i.e., probability) associated with the
  # prediction
  confidence = detections[0, 0, i, 2]

  # filter out weak detections by ensuring the `confidence` is
  # greater than the minimum confidence
  if confidence > args["confidence"]:
    # compute the (x, y)-coordinates of the bounding box for the
    # object
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    location.append((startX, startY, endX, endY))
    # get the rectangle
    X = endX - startX
    Y = endY - startY
    half_lenth = max(X,Y) // 2
    if X > Y:
      middle = (endY + startY) // 2
      startY = max(0, middle - half_lenth)
      endY = min(h, middle + half_lenth)
      
    else:
      middle = (endX + startX) // 2
      startX = max(0, middle - half_lenth)
      endX = min(w, middle + half_lenth)
      
    crop_img = image[startY:endY, startX:endX]
    crop_img = cv2.resize(crop_img, (64, 64))
    crop_img = np.asarray(crop_img, dtype=np.float32) / 255
    faces.append(crop_img)

    

    # # draw the bounding box of the face along with the associated
		# # probability
		# text = "{:.2f}%".format(confidence * 100)
		# y = startY - 10 if startY - 10 > 10 else startY + 10

		# cv2.rectangle(image, (startX, startY), (endX, endY),
		# 	(0, 0, 255), 2)
		# cv2.putText(image, text, (startX, y),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

result = prediction(faces)
print(result)
for i in range(len(result)):
  if result[i] == 1:
    text = 'female'
  else:
    text = 'male'

  (startX, startY, endX, endY) = location[i]
  y = startY - 10 if startY - 10 > 10 else startY + 10
  cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
  cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		
		
		

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)





