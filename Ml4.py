# import the required libraries

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("breast-cancer  dataset.csv")
print(df.head(5))

# Check for missing values:
print(df.isnull().sum())

# Understand data distribution:
print(df.describe())

# Check class distribution: (like 'diagnosis' columns)
print(df["diagnosis"].value_counts())

# Convert categorical label into numerical value (eg, Malignant = 1, Benign = 0):
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
df['diagnosis'].head(5)

# Split dataset

from sklearn.model_selection import train_test_split
x = df.drop(['id','diagnosis'],axis = 1)
y = df['diagnosis']
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.2, random_state=42)

# Train the model using Support Vector Machines

from sklearn.svm import SVC
model = SVC(kernel='rbf')
model.fit(x_train, y_train)


# Evaluate

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))