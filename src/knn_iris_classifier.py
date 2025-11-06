# Importing essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning modules from scikit-learn
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset from a CSV file
df = pd.read_csv('src/iris.csv')
df.head()

# Mapping string species names to integer values
# This step converts categorical labels into numeric form for the model
species_mapping = {
    'Iris-setosa': 1,
    'Iris-versicolor': 2,
    'Iris-virginica': 3
}

# Apply the mapping and ensure the column is of integer type
df['species'] = df['species'].replace(species_mapping).astype(int)
df.head()

# Separate features (X) and target labels (y)
X = df.drop('species', axis=1)
X.head()

y = df['species']
y.value_counts()

# Split the dataset into training and testing subsets
# test_size=0.2 means 20% of the data is used for testing
# shuffle=True randomizes the data before splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Create the K-Nearest Neighbors classifier
knn_model = KNeighborsClassifier()

# Train (fit) the model using the training data
knn_model.fit(X_train, y_train)

# Predict on the training data
y_pred_train = knn_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)

# Predict on the test data
y_pred_test = knn_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Display initial accuracy before scaling
print(f"Accuracy\tTrain = {train_accuracy:.2%}\tTest = {test_accuracy:.2%}")

# Standardize the feature values (mean=0, std=1)
# This is crucial for distance-based models like KNN
scaler = StandardScaler()

# Fit the scaler on the training data and apply the transformation
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Retrain the KNN model with scaled data
knn_model.fit(X_train, y_train)

# Recalculate training and testing accuracies
y_pred_train = knn_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)

y_pred_test = knn_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Display updated accuracies after scaling
print(f"Accuracy\tTrain = {train_accuracy:.2%}\tTest = {test_accuracy:.2%}")

# Create an inverse mapping for displaying results in text
inverse_mapping = {value: key for key, value in species_mapping.items()}

# --- USER INPUT SECTION ---
# Collecting new sample data from the user
# Each feature must be entered manually

sepal_length = float(input("Sepal Length (cm): "))
sepal_width  = float(input("Sepal Width (cm): "))
petal_length = float(input("Petal Length (cm): "))
petal_width  = float(input("Petal Width (cm): "))

# Get feature names used during scaling (ensures correct column order)
feature_names = scaler.feature_names_in_

# Create a DataFrame for the user's input data
user_data = pd.DataFrame(
    data=[[sepal_length, sepal_width, petal_length, petal_width]], 
    columns=feature_names
)

# Apply the same scaling used for the training data
user_data = scaler.transform(user_data)

# Predict the species using the trained model
prediction = knn_model.predict(user_data)
result = prediction[0]

# Convert numeric result back to species name
species_name = inverse_mapping[result]

# Display the final prediction result
print("---  Classification Result ---")
print(f"The model predicted the species as: **{species_name}** (Class {result})")