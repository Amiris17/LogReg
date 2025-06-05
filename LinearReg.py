# Amiris Olivo
# Linear Regression Project usg Ecommerce.csv 
# https://www.kaggle.com/datasets/kolawale/focusg-on-mobile-app-or-website


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ecommerce.csv")

df.head()
df.info()





# EDA

sns.jotplot(x="Time on Website", y="Yearly Amount Spent", data=df, alpha=0.5)




sns.jotplot(x="Time on App", y="Yearly Amount Spent", data=df, alpha=0.5)





sns.pairplot(df, kd='scatter',plot_kws={'alpha': 0.4})


 [90]:


sns.lmplot(x='Length of Membership', y='Yearly Amount Spent',
           data=df,
           scatter_kws={'alpha':0.3})



from sklearn.model_selection import tra_test_split


X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']


# [93]:


X_train, X_test, y_train, y_test =train_test_split(X,y, test_size= 0.3, random_state=42)





X_train




# Train the Model





from sklearn.lear_model import LinearRegression





linear_model=LinearRegression()





linear_model.fit(X_tra, y_tra)




linear_model.coef_





cdf= pd.DataFrame(linear_model.coef_, X.columns, columns=['Coef'])
print (cdf)




#Coefficients are the weight between 




# predictions





predictions=lear_model.predict(X_test)
predictions





sns.scatterplot(x=predictions,y=y_test)
plt.xlabel("Predictions")
plt.title("Evaluation of lear regression Model")





from sklearn.metrics import mean_squared_error, mean_absolute_error
import math





prt("Mean Absolute Erorr:" , mean_absolute_error(y_test, predictions))
prt ("Mean Squared Error:", mean_squared_error(y_test,predictions))
prt("RMSE:",math.sqrt(mean_squared_error(y_test,predictions)))





# residuals





residuals= y_test- predictions




sns.displot(residuals, bs=50,kde=True)


import pylab
import scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=pylab)

