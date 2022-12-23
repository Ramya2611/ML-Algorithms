#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#import dataset
companies = pd.read_csv('D:/ML- Data Science/Linear Regression/1000_Companies.csv')
X = companies.iloc[:, :-1].values
y = companies.iloc[:, 4].values


# In[3]:


companies.head()


# In[4]:


#Data Visualisation
#Building correlation matrix

sns.heatmap(companies.corr())


# In[5]:


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

onehotencoder = ColumnTransformer([("State",OneHotEncoder(),[3])],remainder = 'passthrough')

X = onehotencoder.fit_transform(X)

X = X[:, 1:]
print(X[0])



# In[6]:


#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[7]:


#Fitting multiple Linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[8]:


#predicting the test results
y_pred = regressor.predict(X_test)
print(y_pred)


# In[10]:


#Calculating the coefficients
print(regressor.coef_)


# In[11]:


#Calculating Intercepts
print(regressor.intercept_)


# In[12]:


#Calculating the R squared value
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[ ]:




