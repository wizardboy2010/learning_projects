import numpy as np
import pickle
import collections
import tensorflow as tf
import matplotlib.pyplot as plt

def loaddata(fpath):
    with open(fpath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    return d['data'], d['labels']

d1, l1 = loaddata('cifar-10-batches-py/data_batch_1')
d2, l2 = loaddata('cifar-10-batches-py/data_batch_2')
d3, l3 = loaddata('cifar-10-batches-py/data_batch_2')
d4, l4 = loaddata('cifar-10-batches-py/data_batch_2')
d5, l5 = loaddata('cifar-10-batches-py/data_batch_2')

img = np.vstack([d1, d2, d3, d4, d5])
img = img.reshape(-1, 32, 32, 3).astype(np.float32)
del d1, d2, d3, d4, d5
lab = np.concatenate([np.array(l1), np.array(l2), np.array(l3), np.array(l4), np.array(l5)])
del l1, l2, l3, l4, l5


# 1. airplane  
# 2. automobile  
# 3. bird  
# 4. cat  
# 5. deer  
# 6. dog  
# 7. frog  
# 8. horse  
# 9. ship  
# 10. truck 

class getdata(object):
    
    def __init__(self, images, labels, one_hot = True):
        assert images.shape[0] == labels.shape[0]
        self.num_examples = images.shape[0]
        assert images.shape[3] == 3
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

Datasets = collections.namedtuple('Datasets', ['train', 'test'])

def createdata(images, labels):
    perm = np.arange(images.shape[0])
    np.random.shuffle(perm)
    train_images = images[perm]
    train_labels = labels[perm]
    validation_images = images[int(0.7*train_images.shape[0]):]
    validation_labels = labels[int(0.7*train_labels.shape[0]):]
    train_images = images[:int(0.7*train_images.shape[0])]
    train_labels = labels[:int(0.7*train_labels.shape[0])]
    
    train = getdata(train_images, train_labels, one_hot = True)
    validation = getdata(validation_images, validation_labels, one_hot = True)
    #test = getdata(test_images, test_labels, one_hot = True)
    
    return Datasets(train=train, test=validation)

data = createdata(img, lab)


tf.logging.set_verbosity(tf.logging.INFO)

def model(features, labels, mode):

    input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])
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
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name = "Fc1")

    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10, name = "out")

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
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], 
                                                       name = "Accuracy_value")}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


classifier = tf.estimator.Estimator(model_fn=model, model_dir="tboard/try1")

train_fn = tf.estimator.inputs.numpy_input_fn(x={"x": data.train.images}, 
                                                    y=data.train.labels, 
                                                    batch_size=100,
                                                    num_epochs=None,
                                                    shuffle=True)


type(data.train.images[1,1,1,1])

classifier.train(input_fn=train_fn,steps=20000)

eval_fn = tf.estimator.inputs.numpy_input_fn(x={"x": data.train.images},
                                                        y=data.train.labels,
                                                        num_epochs=1,
                                                        shuffle=False)
eval_results = classifier.evaluate(input_fn=eval_fn)
print(eval_results)
