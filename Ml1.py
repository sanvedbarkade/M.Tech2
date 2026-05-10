import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('real_estate_price_size - slr.csv')
data.head()

data.describe()
data.info()

x = data['size']
y = data['price']


plt.scatter(x,y)
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10)

x_train.shape

x_matrix = x_train.values.reshape(-1,1)

x_matrix.shape

reg = LinearRegression()
reg.fit(x_matrix,y_train)

m = reg.coef_
print(m)

c = reg.intercept_
print(c)

reg.score(x_matrix,y_train)

axes = plt.axes()
axes.scatter(x,y)

axes.plot(x, m*x+c)
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()

# Predit the price of an apartment of area (size) 950 sq. ft

a = pd.Series([950])

apred = a.values.reshape(-1,1)

reg.predict(apred)

a.shape

x_test_matrix = x_test.values.reshape(-1,1)

x_test_matrix.shape


y_pred_test = reg.predict(x_test_matrix)

y_pred_test

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred_test)

print(mse)


score = r2_score(y_test, y_pred_test)

print(score)