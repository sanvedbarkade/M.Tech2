import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.linear_model import Perceptron 
from sklearn.metrics import accuracy_score, confusion_matrix 
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris 
import pandas as pd 
# Load the dataset 
iris = load_iris() 
# Access data (X) and target (y) 
X = iris.data 
y = iris.target 
# Optional: Load as a pandas DataFrame 
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names) 
iris_df['species'] = iris.target 
print(iris_df.head()) 
data = iris_df.dropna() 
data = iris_df.drop_duplicates() 
data = data[data['species'] < 2] 
X = data.drop('species', axis=1)

y = data['species'] 
scaler = StandardScaler() 
X = scaler.fit(X).transform(X) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 
perceptron = Perceptron(max_iter = 40, eta0=0.1, random_state=0) 
perceptron.fit(X_train, y_train) 
model = perceptron.fit(X_train, y_train) 
y_pred = perceptron.predict(X_test) 
print(y_pred) #predicted outputs 
print(y_test) #actual outputs 
accuracy = accuracy_score(y_test, y_pred) 
print("accuracy = ",accuracy*100) 
conf = confusion_matrix(y_test, y_pred) 
print(conf) 
sns.heatmap(conf, annot=True, cmap='Blues') 
plt.xlabel('Predicted') 
plt.ylabel('Actual') 
plt.show() 