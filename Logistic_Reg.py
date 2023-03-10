#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
digits = load_digits()


# In[2]:


print("Image Data Shape", digits.data.shape)
print("Label Data Shape", digits.target.shape)


# In[5]:


import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8,8)),cmap=plt.cm.gray)
    plt.title('Training: %i\n' %label, fontsize = 20)


# In[6]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.23, random_state = 2)


# In[7]:


print(x_train.shape)


# In[8]:


print(y_train.shape)


# In[9]:


print(x_test.shape)


# In[10]:


print(y_test.shape)


# In[11]:


from sklearn.linear_model import LogisticRegression


# In[14]:


logisticRegr = LogisticRegression(solver='lbfgs', max_iter=10000)
logisticRegr.fit(x_train, y_train)


# In[15]:


#Returns a numpy array
#Predict for one observation (image)

print(logisticRegr.predict(x_test[0].reshape(1, -1)))


# In[16]:


#for next 10 numbers
logisticRegr.predict(x_test[0:10])


# In[17]:


#for all numbers
predictions = logisticRegr.predict(x_test)


# In[18]:


score = logisticRegr.score(x_test,y_test)
print(score)


# In[19]:


#try predictions in confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


# In[20]:


cm = metrics.confusion_matrix(y_test, predictions)
print(cm)


# In[22]:


plt.figure(figsize = (9,9))
sns.heatmap(cm, annot=True, fmt = ".3f", linewidths = .5, square = True, cmap = 'Blues_r' );
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# In[23]:


index = 0
classifiedIndex = []
for predict, actual in zip(predictions, y_test):
    if predict == actual:
        classifiedIndex.append(index)
    index +=1
plt.figure(figsize=(20,3))
for plotIndex, wrong in enumerate(classifiedIndex[0:4]):
    plt.subplot(1,4, plotIndex +1)
    plt.imshow(np.reshape(x_test[wrong], (8,8)), cmap=plt.cm.gray)
    plt.title("Predicted: {}, Actual: {}" .format(predictions[wrong], y_test[wrong]), fontsize = 20)
    


# In[ ]:




