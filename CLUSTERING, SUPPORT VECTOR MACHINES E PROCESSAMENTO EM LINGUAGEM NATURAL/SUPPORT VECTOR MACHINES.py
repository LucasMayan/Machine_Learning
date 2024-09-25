# Importando as bibliotecas necessárias
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Carregando o conjunto de dados Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronizando os dados (SVM é sensível à escala dos dados)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinando o modelo SVM
svm_model = SVC(kernel='linear')  # Você pode experimentar outros kernels, como 'rbf' ou 'poly'
svm_model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = svm_model.predict(X_test)

# Avaliando o modelo
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
