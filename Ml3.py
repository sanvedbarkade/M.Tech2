# Import Libraries:
# Import pandas for data handling, train_test_split for splitting data,
# DecisionTreeClassifier for training the model, and accuracy_score for evaluation from sklearn.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load and Prepare Data
df = pd.read_csv('Admission dataset.csv')
X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']]
y = df['Chance of Admit'] > 0.5 # Example for binary classification (Admitted if Chance > 0.5)

df.shape
df.head()
# Split Data: Divide your dataset into training and testing sets to evaluate the model's performance on unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Model: Initialize and train a DecisionTreeClassifier on your training data.
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make Predictions and Evaluate: Use the trained model to make predictions on the test set and evaluate its accuracy.
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Predict for New Data : Predict the admission chance for a new, unseen student profile.

new_student_data = pd.DataFrame([[300, 115, 4, 4.5, 4.0, 9.1, 1]],
columns=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research'])
new_prediction = model.predict(new_student_data)
print(f"Prediction for new student: {'Admitted' if new_prediction[0] else 'Not Admitted'}")
