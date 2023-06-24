#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# # 1)Glass Data

# In[2]:


glass=pd.read_csv(r"C:\Users\sagar\Desktop\sagar\sagar_assignment\Assignment13\glass.csv")
glass


# In[3]:


glass.describe()


# In[4]:


glass.shape


# In[5]:


glass.info()


# In[6]:


array=glass.values


# In[7]:


x=array[:,0:9]


# In[8]:


x


# In[9]:


y=array[:,9]


# In[10]:


y


# In[11]:


num_folds=10


# In[12]:


kfold=KFold(num_folds)


# In[13]:


model=KNeighborsClassifier(n_neighbors=15)


# In[14]:


result=cross_val_score(model,x,y,cv=kfold)


# In[15]:


glasskfold=result.mean()*100


# In[16]:


glasskfold


# In[17]:


from sklearn.model_selection import GridSearchCV
import numpy


# In[18]:


n_neighbors=numpy.array(range(1,30))


# In[19]:


n_neighbors


# In[20]:


param_grid=dict(n_neighbors=n_neighbors)


# In[21]:


model1=KNeighborsClassifier()


# In[22]:


grid=GridSearchCV(estimator=model1,param_grid=param_grid)


# In[23]:


grid.fit(x,y)


# In[24]:


glassgrid=(grid.best_score_)*100


# In[25]:


glassgrid


# In[26]:


print(grid.best_params_)


# In[27]:


import matplotlib.pyplot as plt


# In[28]:


k_range=range(1,50)
k_scores=[]


# In[29]:


for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    score=cross_val_score(knn,x,y,cv=5)
    k_scores.append(score.mean())


# In[30]:


plt.plot(k_range,k_scores)
plt.xlabel("K value")
plt.ylabel("score")
plt.show()


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.33, random_state=30)


# In[33]:


model=KNeighborsClassifier()
model.fit(x_train,y_train)


# In[34]:


result=model.score(x_test,y_test)


# In[35]:


glasstrain=result.mean()*100


# In[36]:


glasstrain


# In[37]:


d1={'Model validation':['KFold','GridSearch','Train_Test_split'],'Acuracy':[glasskfold,glassgrid,glasstrain]}
KNeighbors_Frame=pd.DataFrame(d1)
KNeighbors_Frame


# # 2) Zoo Data

# In[38]:


zoo=pd.read_csv(r"C:\Users\sagar\Desktop\sagar\sagar_assignment\Assignment13\Zoo.csv")
zoo.head()


# In[39]:


zoo.describe()


# In[40]:


zoo.info()


# In[41]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
zoo.iloc[:,0]=labelencoder.fit_transform(zoo.iloc[:,0])


# In[42]:


zoo.head()
zoo.shape


# In[43]:


x=zoo.iloc[:,0:17]
y=zoo.iloc[:,17]


# In[44]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)


# In[45]:


model2=KNeighborsClassifier()
model2.fit(x_train,y_train)


# In[46]:


result=model2.score(x_test,y_test)
train_test=result.mean()*100
train_test


# In[47]:


kfold1=KFold(n_splits=5)
model3=KNeighborsClassifier(n_neighbors=10)
result3=cross_val_score(model3,x,y,cv=kfold1)
zookfold=result.mean()*100
zookfold


# In[48]:


n_neighbors3=numpy.array(range(1,40))
param_grid1=dict(n_neighbors=n_neighbors3)
model4=KNeighborsClassifier()
grid4=GridSearchCV(estimator=model4,param_grid=param_grid1)


# In[49]:


grid4.fit(x,y)

GridSearch=(grid4.best_score_)*100



# In[50]:


GridSearch


# In[51]:


print(grid4.best_params_)


# In[52]:


d2={'Model validation':['Train_Test','KFold','GridSearch'],'Accuracy':[train_test,zookfold,GridSearch]}
ZooKneighborsClassification=pd.DataFrame(d2)
ZooKneighborsClassification


# In[ ]:




