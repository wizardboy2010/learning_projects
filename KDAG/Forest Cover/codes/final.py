
# coding: utf-8

# # Forest Cover Prediction - Kaggle

# ## Data Importing

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')


# In[2]:


data = pd.read_csv('train.csv')
data = data.drop('Id', 1)


# ## Statistical description

# In[3]:


#######  Gives Number of missing values in each feature
data.isnull().sum()


# * No Missing Data

# In[4]:


####### Describe dataset
pd.set_option('display.max_columns', None)
data.describe()


# * ##### Removing unwanted fearures

# In[5]:


useless = []
### if data is constant then remove it
for i in data.columns:
    if data[i].std() == 0:
        useless.append(i)
print(useless)
data.drop(useless, 1, inplace = True)


# ## Skewness

# In[6]:


###### Gives amount of Skewness in the data
data.skew()


# * Soil_Type8 , Soil_Type25 are highly skew

# In[7]:


#####  Shows number of datapoints in each class
data.groupby('Cover_Type').size()


# * All classes are equally present

# ## Correlation Analysis

# #### Heat Map of Non-Categorical Data

# In[8]:


plt.subplots(figsize=(10,10))
sns.heatmap(data[['Elevation','Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points','Cover_Type']].corr(), annot=True)


# #### Highly Correlated features

# In[9]:


limit = 0.5
corr = data.corr()
corr_list = []
for i in range(10):
    for j in range(i+1, 10):
        if 1 > corr.iloc[i,j] >= limit or 0 > corr.iloc[i,j] <= -limit:
            corr_list.append([corr.iloc[i,j], i, j])


# #### PairPlot of Highly Correlated features

# In[10]:


for x, y, z in corr_list:
    sns.pairplot(data, hue = "Cover_Type", size = 4, x_vars = data.columns[y], y_vars = data.columns[z])


# ## Preparing Data

# In[11]:


x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


# In[12]:


#### divide data into train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# In[13]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train[:,:10] = sc_x.fit_transform(x_train[:,:10])
x_test[:,:10] = sc_x.transform(x_test[:,:10])


# ## Models

# > ### Naive Bayes

# In[14]:


from sklearn.naive_bayes import GaussianNB
model1 = GaussianNB()
model1.fit(x_train, y_train)
print(model1.score(x_train, y_train))
print(model1.score(x_test, y_test))


# In[15]:


from sklearn.naive_bayes import BernoulliNB
model2 = BernoulliNB(alpha = 0.8)
model2.fit(x_train, y_train)
print(model2.score(x_train, y_train))
print(model2.score(x_test, y_test))


# * BernoulliNB gave good results compared to GaussianNB

# > ### Decision Tree

# In[16]:


from sklearn import tree
model3 = tree.DecisionTreeClassifier(min_samples_split = 2,random_state = 42, max_depth = 50, max_leaf_nodes = 900)
model3.fit(x_train, y_train)
print(model3.score(x_train, y_train))
print(model3.score(x_test, y_test))


# > ### Random Forest

# In[17]:


from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier(n_estimators= 100, min_samples_split=2, n_jobs = 4, max_leaf_nodes = 850, random_state = 42)
model4.fit(x_train, y_train)
print(model4.score(x_train, y_train))
print(model4.score(x_test, y_test))


# > ### AdaBoost Classifier

# In[18]:


from sklearn.ensemble import AdaBoostClassifier
model5 = AdaBoostClassifier(n_estimators = 6, learning_rate = 1, random_state = 42)
model5.fit(x_train, y_train)
print(model5.score(x_train, y_train))
print(model5.score(x_test, y_test))


# > ### Gradient Boosting

# In[19]:


from sklearn.ensemble import GradientBoostingClassifier
model6 = GradientBoostingClassifier(learning_rate = 0.5, n_estimators = 100, max_depth = 4, random_state = 42)
model6.fit(x_train, y_train)
print(model6.score(x_train, y_train))
print(model6.score(x_test, y_test))


# * RandomForest gave good results compared to all other classifiers.

# ## Submission in Kaggle

# In[20]:


data1 = pd.read_csv('test.csv')
data1 = data1.drop('Id', 1)
data1.drop(useless, 1, inplace = True)

tx = data1.iloc[:,:].values
tx[:,:10] = sc_x.transform(tx[:,:10])

ty = model4.predict(tx)


# In[21]:


sub = pd.read_csv('test.csv').Id
sub = pd.DataFrame(sub)
sub['Cover_Type'] = ty

sub.to_csv('sub2.csv', index = False)

