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

# ==========================
# 1. Load and Inspect Data
# ==========================

# Load the Iris dataset from a CSV file
df = pd.read_csv('src/iris.csv')
print("\nDataset loaded successfully!\n")
print(df.head())

# ==========================
# 2. Data Exploration (Visualization)
# ==========================

print("\nVisualizing feature relationships and correlations...\n")

# Pairplot: visualize how the features are related
sns.pairplot(df, hue='species', diag_kind='kde')
plt.suptitle("Pairwise Feature Relationships in Iris Dataset", y=1.02)
plt.show()

# Heatmap: correlation between numerical features
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# ==========================
# 3. Data Preprocessing
# ==========================

# Mapping string species names to integer values
# This step converts categorical labels into numeric form for the model
species_mapping = {
    'Iris-setosa': 1,
    'Iris-versicolor': 2,
    'Iris-virginica': 3
}

# Apply the mapping and ensure the column is of integer type
df['species'] = df['species'].replace(species_mapping).astype(int)

# Separate features (X) and target labels (y)
X = df.drop('species', axis=1)
y = df['species']

print("\nClass distribution:")
print(y.value_counts())

# ==========================
# 4. Train/Test Split
# ==========================

# Split the dataset into training and testing subsets
# test_size=0.2 means 20% of the data is used for testing
# shuffle=True randomizes the data before splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# ==========================
# 5. Initial Model Training (without scaling)
# ==========================

# Create the K-Nearest Neighbors classifier
knn_model = KNeighborsClassifier()

# Train (fit) the model using the training data
knn_model.fit(X_train, y_train)

# Predict on the training and testing data
y_pred_train = knn_model.predict(X_train)
y_pred_test = knn_model.predict(X_test)

# Calculate accuracies
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nInitial Accuracy (before scaling):")
print(f"Accuracy\tTrain = {train_accuracy:.2%}\tTest = {test_accuracy:.2%}")

# ==========================
# 6. Feature Scaling
# ==========================

# Standardize the feature values (mean=0, std=1)
# This is crucial for distance-based models like KNN
scaler = StandardScaler()

# Fit the scaler on the training data and apply the transformation
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# 7. Retrain the Model (after scaling)
# ==========================

knn_model.fit(X_train, y_train)

# Recalculate training and testing accuracies
y_pred_train = knn_model.predict(X_train)
y_pred_test = knn_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nAccuracy after feature scaling:")
print(f"Accuracy\tTrain = {train_accuracy:.2%}\tTest = {test_accuracy:.2%}")

# ==========================
# 8. Visualizing Model Performance
# ==========================

print("\nVisualizing model performance...\n")

accuracies = [train_accuracy, test_accuracy]
labels = ['Train', 'Test']

plt.bar(labels, accuracies, color=['skyblue', 'orange'])
plt.title("KNN Accuracy Comparison (After Scaling)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# ==========================
# 9. Demonstrating KNN Internals (with NumPy)
# ==========================

# Example: calculate Euclidean distance manually between two samples
print("\nExample of manual Euclidean distance (KNN principle):")

sample_1 = X.iloc[0].values
sample_2 = X.iloc[50].values
distance = np.sqrt(np.sum((sample_1 - sample_2) ** 2))
print(f"Distance between sample 1 and sample 50: {distance:.3f}\n")

# ==========================
# 10. User Input for Prediction
# ==========================

# Create an inverse mapping for displaying results in text
inverse_mapping = {value: key for key, value in species_mapping.items()}

# --- USER INPUT SECTION ---
# Collecting new sample data from the user
# Each feature must be entered manually
print("Enter the flower measurements to predict its species:\n")

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
print("\n--- Classification Result ---")
print(f"The model predicted the species as: **{species_name}** (Class {result})\n")