import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
data = data.drop('Id', 1)
data['Hillshade_9am'] = data['Hillshade_9am']/255
data['Hillshade_Noon'] = data['Hillshade_Noon']/255
data['Hillshade_3pm'] = data['Hillshade_3pm']/255


x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y.reshape(-1,1)).toarray()

#from sklearn.cross_validation import train_test_split        #cross_validation is going to be removed in future version of library
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train[:,[0,1,2,3,4,5,9]] = sc_x.fit_transform(x_train[:,[0,1,2,3,4,5,9]])
x_test[:,[0,1,2,3,4,5,9]] = sc_x.transform(x_test[:,[0,1,2,3,4,5,9]])

import tensorflow as tf
sess = tf.InteractiveSession()

X = tf.placeholder(tf.float64, [None, 54])
Y_ = tf.placeholder(tf.float64, [None, 7])
# variable learning rate
lr = tf.placeholder(tf.float64)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float64)

L = 100
M = 40

# Weights initialised with small random values between -0.2 and +0.2
W1 = tf.Variable(tf.truncated_normal([54, L], stddev=0.1, dtype = tf.float64))
B1 = tf.Variable(tf.truncated_normal([L], stddev=0.1, dtype = tf.float64))
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1, dtype = tf.float64))
B2 = tf.Variable(tf.truncated_normal([M], stddev=0.1, dtype = tf.float64))
W3 = tf.Variable(tf.truncated_normal([M, 7], stddev=0.1, dtype = tf.float64))
B3 = tf.Variable(tf.truncated_normal([7], stddev=0.1, dtype = tf.float64))

Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)

Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)

Ylogits = tf.matmul(Y2d, W3) + B3
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

#initialize tf variables and session 
sess.run(tf.global_variables_initializer())

train_cost = []
test_cost = []
train_acc = []
test_acc = []

for i in range(50):  
   train_step.run(feed_dict={X: x_train, Y_: y_train, lr: 0.008, pkeep: 1})
   trc = cross_entropy.eval(feed_dict={X: x_train, Y_: y_train, pkeep: 1})
   tec = cross_entropy.eval(feed_dict={X: x_test, Y_: y_test, pkeep: 1})
   tra = accuracy.eval(feed_dict={X: x_train, Y_: y_train, pkeep: 1})
   tea = accuracy.eval(feed_dict={X: x_test, Y_: y_test, pkeep: 1})
   train_cost.append(trc)
   test_cost.append(tec)
   train_acc.append(tra)
   test_acc.append(tea)
   print("For epoch ", i+1, ",Train cost:", trc, ",test cost:", tec,",train acc:", tra,",test acc:", tea)

plt.plot(train_cost, 'blue')
plt.plot(test_cost, 'red')
plt.show()
plt.plot(train_acc, 'blue')
plt.plot(test_acc, 'red')
plt.show()

print("After optimization : ")
training_cost = cross_entropy.eval(feed_dict={X: x_train, Y_: y_train, pkeep: 1})
print("Training cost= ", training_cost)
training_acc = accuracy.eval(feed_dict={X: x_train, Y_: y_train, pkeep: 1})
print("Training accuracy= ", training_acc)
test_acc = accuracy.eval(feed_dict={X: x_test, Y_: y_test, pkeep: 1})
print("test accuracy : ", test_acc)

