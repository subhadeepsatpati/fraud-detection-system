# Fraud Detection System Using Machine Learning

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load Dataset
df = pd.read_csv('dataset/creditcard.csv')

# Show First 5 Rows
print("First 5 Rows:")
print(df.head())

# Dataset Shape
print("\nDataset Shape:")
print(df.shape)

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Count Transactions
print("\nTransaction Counts:")
print(df['Class'].value_counts())

# Separate Fraud and Normal Transactions
fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0]

# Handle Imbalanced Data
normal_sample = normal.sample(n=len(fraud), random_state=42)

# Create Balanced Dataset
new_df = pd.concat([fraud, normal_sample], axis=0)

# Shuffle Dataset
new_df = new_df.sample(frac=1, random_state=42)

print("\nBalanced Dataset Counts:")
print(new_df['Class'].value_counts())

# Split Features and Target
X = new_df.drop(columns='Class', axis=1)
Y = new_df['Class']

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

# Scale Data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Model
model = LogisticRegression(
    max_iter=5000,
    solver='liblinear'
)

# Train Model
model.fit(X_train, Y_train)

# Predict Training Data
X_train_prediction = model.predict(X_train)

# Training Accuracy
training_accuracy = accuracy_score(
    X_train_prediction,
    Y_train
)

print("\nTraining Accuracy:")
print(training_accuracy)

# Predict Test Data
X_test_prediction = model.predict(X_test)

# Test Accuracy
test_accuracy = accuracy_score(
    X_test_prediction,
    Y_test
)

print("\nTest Accuracy:")
print(test_accuracy)

# Confusion Matrix
cm = confusion_matrix(Y_test, X_test_prediction)

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(Y_test, X_test_prediction))

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()