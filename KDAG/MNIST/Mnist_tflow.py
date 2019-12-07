import numpy as np
import collections
import os
import struct
import gzip
import numpy as np
import tensorflow as tf
import urllib
import sys
import matplotlib.pyplot as plt

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def maybe_download(filename):
    WORK_DIRECTORY = os.getcwd()
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

IMAGE_SIZE = 28
def extract_data(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data

def extract_labels(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

train_images = extract_data('train-images-idx3-ubyte.gz', 60000)
train_labels = extract_labels('train-labels-idx1-ubyte.gz', 60000)
test_images = extract_data('t10k-images-idx3-ubyte.gz', 10000)
test_labels = extract_labels('t10k-labels-idx1-ubyte.gz', 10000)

def onehot(labels):
    l = np.zeros([labels.shape[0], 10])
    for i, val in enumerate(labels):
        l[i, val] = 1
    return l

class getdata(object):
    
    def __init__(self, images, labels, one_hot = True):
        assert images.shape[0] == labels.shape[0]
        self.num_examples = images.shape[0]
        assert images.shape[3] == 1
        self.images = images
        self.one_hot = one_hot
        self.labels = labels
        if not one_hot:
            self.labels = onehot(labels)
        self.epoch = 0
        self.startofepoch = 0
        
    def nextbatch(self, batch_size):
        start = self.startofepoch
        end = start + batch_size
        
        if end > self.num_examples:
            self.epoch += 1
            
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            
            start = 0
            end = batch_size
            
        self.startofepoch = end
        return self.images[start:end], self.labels[start:end]

def createdata(train_images, train_labels, test_images, test_labels):
    perm = np.arange(train_images.shape[0])
    np.random.shuffle(perm)
    train_images = train_images[perm]
    train_labels = train_labels[perm]
    validation_images = train_images[int(0.7*train_images.shape[0]):]
    validation_labels = train_labels[int(0.7*train_labels.shape[0]):]
    train_images = train_images[:int(0.7*train_images.shape[0])]
    train_labels = train_labels[:int(0.7*train_labels.shape[0])]
    
    train = getdata(train_images, train_labels, one_hot = True)
    validation = getdata(validation_images, validation_labels, one_hot = True)
    test = getdata(test_images, test_labels, one_hot = True)
    
    return Datasets(train=train, validation=validation, test=test)

data = createdata(train_images, train_labels, test_images, test_labels)

##########################################################################################################################################

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def model(features, labels, mode):

    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name = "Conv1")

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name = "Pool1")
    conv2 = tf.layers.conv2d(inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name = "Conv2")

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name = "Pool2")
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name = "Fc1")

    #dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dense, units=10, name = "out")

    predictions = {
        "classes": tf.argmax(input=logits, axis=1, name = "Prediction"),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name = "Accuracy_value")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name = "Accuracy_value")}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

###########################################################################################################################################

classifier = tf.estimator.Estimator(model_fn=model, model_dir="tflow4/temp2")

# Train the model
train_fn = tf.estimator.inputs.numpy_input_fn(x={"x": data.train.images}, 
                                                    y=data.train.labels, 
                                                    batch_size=100,
                                                    num_epochs=None,
                                                    shuffle=True)
classifier.train(input_fn=train_fn,steps=20000)

# Evaluate the model and print results
eval_fn = tf.estimator.inputs.numpy_input_fn(x={"x": data.validation.images},
                                                        y=data.validation.labels,
                                                        num_epochs=1,
                                                        shuffle=False)
eval_results = classifier.evaluate(input_fn=eval_fn)
print(eval_results)
