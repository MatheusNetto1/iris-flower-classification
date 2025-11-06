# Iris Flower Classification with K-Nearest Neighbors (KNN)

This project implements an **Iris flower classifier** using the **K-Nearest Neighbors (KNN)** algorithm.  
It focuses on **data preprocessing, visualization, and supervised learning evaluation** with the popular Iris dataset.

---

## Project Overview

The goal is to predict the **species of an Iris flower** based on four measurable features:

- **Sepal Length (cm)**
- **Sepal Width (cm)**
- **Petal Length (cm)**
- **Petal Width (cm)**

The model uses the **KNN algorithm** from `scikit-learn` and is trained on the classic **Iris dataset**, widely used for machine learning education and research.

---

## Project Structure

```
├── src/
│ ├── iris.csv                      # Base dataset
│ ├── knn_iris_classifier.ipynb     # Jupyter Notebook (interactive analysis)
│ └── knn_iris_classifier.py        # Python script version (CLI execution)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Requirements & Installation

### Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Run with Jupyter Notebook

```bash
jupyter notebook src/knn_iris_classifier.ipynb
```

### Run as a Python script

```bash
python src/knn_iris_classifier.py
```

### The script will prompt you to enter the following values manually:
```bash
Sepal Length (cm)
Sepal Width (cm)
Petal Length (cm)
Petal Width (cm)
```
> After that, the model will output the predicted flower species.

---

## Process Overview

1. Load and explore the dataset using `pandas`, `seaborn`, and `matplotlib`
2. Map string species names to numeric values
3. Split the dataset into training and test sets (80% / 20%)
4. Train the initial KNN model
5. Standardize feature values using `StandardScaler`
6. Retrain and evaluate the model on scaled data
7. Predict new samples from user input
8. Visualize correlations among numerical features

---

## Data Visualization

### The project includes exploratory plots and a feature correlation heatmap.

### Example of correlation visualization:

```bash
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
```

---

## Machine Learning Model

* Algorithm: K-Nearest Neighbors (KNN)
* Library: `scikit-learn`
* Metric: Accuracy (`accuracy_score`)
* Preprocessing: Standardization with `StandardScaler`

---

## Interpretation

The KNN algorithm classifies new samples based on their Euclidean distance to existing labeled points.<br>
Feature scaling is crucial so that all attributes contribute equally to the distance calculations.

---

## License

This project is licensed under the [MIT License](https://github.com/MatheusNetto1/iris-flower-classification/blob/main/LICENSE)