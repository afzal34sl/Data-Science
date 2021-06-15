# -*- coding: utf-8 -*-
"""
@author: afzal
"""
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('data.csv')  # load data set
#df.head()
print('df.size: ', df.size)

#print('df.head: ', df.head)
#plt.scatter(df['Col1'], df['Col2'], color='red')
#plt.scatter(X, Y, color='red')
#plt.axis([pd.Series.min(df[X]), pd.Series.max(df[X]), pd.Series.min(df[Y]), pd.Series.max(df[Y])])
#plt.show()



X = df.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
Y = df.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
#y_test.head()
#x_test.head()
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(x_train, y_train)#(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(x_test)  # make predictions
# regression coefficients
print('Coefficients: ', linear_regressor.coef_)
print("Mean squared error:",mean_squared_error(y_test, Y_pred))
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(linear_regressor.score(X, Y)))

plt.scatter(X, Y)
plt.plot(x_test, Y_pred, color='green')
plt.show()

##Predicting scores based on number of hours studied
dataset = pd.read_csv('student_scores.csv')
dataset.shape
dataset.head()
dataset.describe()
dataset.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
plt.scatter(X, y)
plt.plot(X_test, y_pred, color='green')
plt.show()
print(df)
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Variance score: {}'.format(linear_regressor.score(X, y)))