#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[84]:


df = pd.read_csv("ecommerce.csv")


# In[85]:


df.head()


# In[86]:


df.info()


# In[87]:


# EDA

sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=df, alpha=0.5)


# In[88]:


sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=df, alpha=0.5)


# In[89]:


sns.pairplot(df, kind='scatter',plot_kws={'alpha': 0.4})


# In[90]:


sns.lmplot(x='Length of Membership', y='Yearly Amount Spent',
           data=df,
           scatter_kws={'alpha':0.3})



# In[91]:


from sklearn.model_selection import train_test_split


# In[92]:


X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']


# In[93]:


X_train, X_test, y_train, y_test =train_test_split(X,y, test_size= 0.3, random_state=42)


# In[94]:


X_train


# In[95]:


# Training the Model


# In[96]:


from sklearn.linear_model import LinearRegression


# In[97]:


linear_model=LinearRegression()


# In[98]:


linear_model.fit(X_train, y_train)


# In[99]:


linear_model.coef_


# In[100]:


cdf= pd.DataFrame(linear_model.coef_, X.columns, columns=['Coef'])
print (cdf)


# In[101]:


#Coefficients are the weight between 


# In[102]:


# predictions


# In[103]:


predictions=linear_model.predict(X_test)
predictions


# In[104]:


sns.scatterplot(x=predictions,y=y_test)
plt.xlabel("Predictions")
plt.title("Evaluation of linear regression Model")


# In[106]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


# In[107]:


print("Mean Absolute Erorr:" , mean_absolute_error(y_test, predictions))
print ("Mean Squared Error:", mean_squared_error(y_test,predictions))
print("RMSE:",math.sqrt(mean_squared_error(y_test,predictions)))


# In[ ]:


# residuals


# In[110]:


residuals= y_test- predictions


# In[114]:


sns.displot(residuals, bins=50,kde=True)


# In[116]:


import pylab
import scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=pylab)

