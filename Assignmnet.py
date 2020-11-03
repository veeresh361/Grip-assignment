#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


d=pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')


# In[3]:


d.isnull().sum()


# In[4]:


d.head()


# In[5]:


x=d['Hours']
y=d['Scores']


# In[6]:


import seaborn as sns


# In[7]:


sns.jointplot(x,y,kind='reg')


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)


# In[10]:


from sklearn.linear_model import LinearRegression


# In[11]:


clf=LinearRegression()


# In[12]:


x_train=np.array(x_train).reshape(-1,1)


# In[13]:


x_test=np.array(x_test).reshape(-1,1)


# In[14]:


clf.fit(x_train,y_train)


# In[15]:


from sklearn.metrics import mean_squared_error


# In[16]:


y_pred=clf.predict(x_test)


# In[17]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[18]:


y_pred=np.array(y_pred).reshape(-1,1)


# In[20]:


sns.jointplot(x_test,y_pred,kind='reg')


# In[ ]:




