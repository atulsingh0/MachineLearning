
# coding: utf-8

# In[1]:

# import 
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression

get_ipython().magic('matplotlib inline')


# In[2]:

# reading the data- 
# data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
# data.to_csv("data/Advertising.csv")


# In[3]:

# reading the data from local system
data = pd.read_csv("data/Advertising.csv")

data.head()


# In[4]:

# converting first column to index
data = pd.read_csv("data/Advertising.csv", index_col=0)

data.head()


# In[5]:

features = ['TV', 'Radio', 'Newspaper']
response = ['Sales']

X = data[features]
y = data[response]

print(X.head())
print(y.head())


# In[6]:

# plotting the relation between TV, Radio and Newspaper with Sales 
sns.pairplot(data=data, x_vars=features, y_vars=response)


# In[7]:

# increasing the size

# plotting the relation between TV, Radio and Newspaper with Sales 
sns.pairplot(data=data, x_vars=features, y_vars=response, size=7)


# In[8]:

# increasing the size

# plotting the relation between TV, Radio and Newspaper with Sales 
sns.pairplot(data=data, x_vars=features, y_vars=response, size=7, aspect=0.8)


# In[9]:

# increasing the size

# plotting the relation between TV, Radio and Newspaper with Sales 
sns.pairplot(data=data, x_vars=features, y_vars=response, size=7, aspect=0.7, kind='reg')


# ### Now doing the Linear Regression

# In[10]:

# import
from sklearn.cross_validation import train_test_split
# from sklearn.metrics import accuracy_score   # not supported by continuous data so, not used in reg model

X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[11]:

# using linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[12]:

# checking the coefficient of Linear model
print(lr.intercept_)
print(lr.coef_)


# In[13]:

# zip
coeff = zip(features, lr.coef_[0])
coeff

for x in coeff:
    print(x)


# In[14]:

# predict
y_pred = lr.predict(X_test)


# ### Choosing Best model  - Calculating Errors

# In[15]:

# define true and predicted response values
true = [100, 50, 30, 20]
pred = [90, 50, 50, 30]


# In[16]:

# calculate MAE by hand
print((10 + 0 + 20 + 10)/4.)

# calculate MAE using scikit-learn
from sklearn import metrics
print(metrics.mean_absolute_error(true, pred))


# In[17]:

# calculate MSE by hand
print((10**2 + 0**2 + 20**2 + 10**2)/4.)

# calculate MSE using scikit-learn
print(metrics.mean_squared_error(true, pred))


# In[18]:

# calculate RMSE by hand
import numpy as np
print(np.sqrt((10**2 + 0**2 + 20**2 + 10**2)/4.))

# calculate RMSE using scikit-learn
print(np.sqrt(metrics.mean_squared_error(true, pred)))


# In[19]:

# Calculating RMSE
np.sqrt(metrics.mean_squared_error(y_test, y_pred))


# ### RMSE help to choose the features

# In[21]:

features = ['TV', 'Radio']
X = data[features]

X_train, X_test, y_train, y_test = train_test_split(X, y)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

np.sqrt(metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:



