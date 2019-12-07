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
'''from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y.reshape(-1,1)).toarray()
'''
#from sklearn.cross_validation import train_test_split        #cross_validation is going to be removed in future version of library
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train[:,[0,1,2,3,4,5,9]] = sc_x.fit_transform(x_train[:,[0,1,2,3,4,5,9]])
x_test[:,[0,1,2,3,4,5,9]] = sc_x.transform(x_test[:,[0,1,2,3,4,5,9]])


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators= 30, min_samples_split=2, n_jobs = 7)
model.fit(x_train, y_train)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))



data1 = pd.read_csv('test.csv')
data1 = data1.drop('Id', 1)
data1['Hillshade_9am'] = data1['Hillshade_9am']/255
data1['Hillshade_Noon'] = data1['Hillshade_Noon']/255
data1['Hillshade_3pm'] = data1['Hillshade_3pm']/255

tx = data1.iloc[:,:].values
tx[:,[0,1,2,3,4,5,9]] = sc_x.transform(tx[:,[0,1,2,3,4,5,9]])

ty = model.predict(tx)

sub = pd.read_csv('test.csv').Id
sub = pd.DataFrame(sub)
sub['Cover_Type'] = ty

sub.to_csv('sub2.csv', index = False)