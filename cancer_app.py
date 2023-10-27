import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score, f1_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('cancer.csv')

# Replace '?' with 999
df['bare_nuclei'].replace('?', 999, inplace=True)
df['bare_nuclei'] = df['bare_nuclei'].astype('int64')

# Split the data into train and test
X = df.drop(['id', 'classes'], axis=1)
y = df['classes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the K-Nearest Neighbors model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Define a Streamlit app
st.title("Breast Cancer Classification")

# Display the first few rows of the dataset
st.subheader("Dataset Overview")
st.write(df.head())

# Display dataset shape
st.subheader("Dataset Shape")
st.write(df.shape)

# Display dataset summary statistics
st.subheader("Summary Statistics")
st.write(df.describe())

# Check for null values
st.subheader("Null Values")
st.write(df.isnull().sum())

# Correlation heatmap
st.subheader("Correlation Heatmap")
plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True)
st.pyplot(plt)

# Model accuracy
st.subheader("Model Accuracy")
accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy Score: ", accuracy)

# Model r2 score
st.subheader("R-squared (r2) Score")
r2 = r2_score(y_test, y_pred)
st.write("R-squared Score: ", r2)

# ROC curve
st.subheader("Receiver Operating Characteristics (ROC) Curve")
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='orange', label='ROC')
ax.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristics (ROC) Curve')
ax.legend()
st.pyplot(fig)

# F1-score
st.subheader("F1 Score")
f1 = f1_score(y_test, y_pred)
st.write("F1 Score: ", f1)
