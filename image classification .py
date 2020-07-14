#!/usr/bin/env python
# coding: utf-8

# In[47]:


#importing dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#I could only find the test and train files seperately so I didnt import train_test_split


# In[48]:


#getting train.csv
dataTrain=pd.read_csv("mnist_train.csv")


# In[57]:


dataTrain.head(2)
#seperating xtrain and ytrain  
x_train=dataTrain.iloc[:,1:].values
y_train=dataTrain.iloc[:,0].values
#checking how the data is represented
a=dataTrain.iloc[4,1:].values
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)
print(y[4])


# In[50]:


#getting test.csv
dataTest=pd.read_csv('mnist_test.csv')
#seperating xtest and ytest
x_test=dataTest.iloc[:,1:]
y_test=dataTest.iloc[:,0]


# In[51]:


#calling classifier
classifier=RandomForestClassifier(n_estimators=100)


# In[52]:


#fiting data into model
classifier.fit(x_train,y_train)


# In[53]:


prediction=classifier.predict(x_test)


# In[54]:


j=0
count=0
for i in prediction:
    if i==y_test[j]:
        count+=1
    j+=1
accuracy=(count/len(y_test))*100
print(accuracy)

