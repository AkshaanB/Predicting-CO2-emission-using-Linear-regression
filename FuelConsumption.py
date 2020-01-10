#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


# In[35]:


url = "D:\My Work\Otherthan Syllabus\Data Science\Projects\FuelConsumption.csv"


# In[36]:


df = pd.read_csv(url,encoding='cp1252')


# In[37]:


df


# In[38]:


df.isnull()


# In[39]:


df.isnull().sum()


# In[41]:


wanted_data = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','CO2EMISSIONS']]


# In[42]:


wanted_data


# In[43]:


wanted_data.isnull().sum()


# In[ ]:





# In[44]:


wanted_data


# In[46]:


plt.scatter(wanted_data.FUELCONSUMPTION_CITY, wanted_data.CO2EMISSIONS,color="blue")
plt.xlabel("FuelConsumption")
plt.ylabel("CO2 Emissions")
plt.show()


# In[47]:


plt.scatter(wanted_data.ENGINESIZE, wanted_data.CO2EMISSIONS,color="blue")
plt.xlabel("EngineSize")
plt.ylabel("CO2 Emissions")
plt.show()


# In[49]:


plt.scatter(wanted_data.CYLINDERS, wanted_data.CO2EMISSIONS,color="blue")
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emissions")
plt.show()


# In[50]:


wanted_data.hist()
plt.show()


# In[51]:


wanted_data


# In[52]:


msk = np.random.rand(len(wanted_data)) < 0.8
train = wanted_data[msk]
test = wanted_data[~msk]


# In[53]:


train


# In[54]:


test


# In[55]:


plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.xlabel('Engine Size')
plt.ylabel('CO2Emissions')
plt.show()


# In[56]:


regression = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regression.fit(train_x,train_y)
print('Coefficients: ',regression.coef_)
print('Intercept: ',regression.intercept_)


# In[57]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regression.coef_[0][0]*train_x + regression.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Co2Emission")


# In[62]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regression.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) **2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y))


# In[66]:


print("Predicted Co2 emission: " , test_y_hat )


# In[ ]:





# In[ ]:




