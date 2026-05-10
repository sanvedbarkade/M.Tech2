import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("real_estate_price_size_year - mlr.csv") 
data.head()

data.describe()

data.info()

X = data[['size','year']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
model = LinearRegression() 
model.fit(X_train, y_train)

m = model.coef_ 
print(m)

c = model.intercept_ 
print(c)


y_pred = model.predict(X_test) 
r2 = r2_score(y_test, y_pred) 
print("R-squared:", r2)

size = float(input("Enter size: ")) 
year = float(input("Enter year: ")) 
prediction = model.predict([[size, year]])
print("Predicted price:", prediction[0])