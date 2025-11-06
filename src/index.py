import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('src/iris.csv')
df.head()

mapeamento_species = {
    'Iris-setosa': 1,
    'Iris-versicolor': 2,
    'Iris-virginica': 3
}

df['species'] = df['species'].replace(mapeamento_species).astype(int)
df.head()

X = df.drop('species', axis=1)
X.head()

y = df['species']
y.value_counts()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True)

knn_model = KNeighborsClassifier()

knn_model.fit(X_train, y_train)

y_pred_train = knn_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)

y_pred_test = knn_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"accuracy\ttrain = {train_accuracy:.2%} \ttest = {test_accuracy:.2%}")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn_model.fit(X_train, y_train)

y_pred_train = knn_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)

y_pred_test = knn_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"accuracy\ttrain = {train_accuracy:.2%} \ttest = {test_accuracy:.2%}")

mapa_inverso = {valor: chave for chave, valor in mapeamento_species.items()}

sepal_length = float(input("Comprimento da Sépala (sepal length): "))
sepal_width  = float(input("Largura da Sépala (sepal width): "))
petal_length = float(input("Comprimento da Pétala (petal length): "))
petal_width  = float(input("Largura da Pétala (petal width): "))

feature_names = scaler.feature_names_in_

dados_usuario = pd.DataFrame(
    data=[[sepal_length, sepal_width, petal_length, petal_width]], 
    columns=feature_names
)

dados_usuario = scaler.transform(dados_usuario)

predicao = knn_model.predict(dados_usuario)
resultado = predicao[0]

nome_especie = mapa_inverso[resultado]

print("---  Resultado da Classificação ---")
print(f"O modelo previu que a espécie é: **{nome_especie}** (Classe {resultado})")