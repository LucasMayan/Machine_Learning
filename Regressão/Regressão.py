import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Gerar um dataset de exemplo
np.random.seed(42)
house_size = np.random.rand(100, 1) * 100  # Tamanhos das casas entre 0 e 100 metros quadrados
house_price = house_size * 3000 + np.random.randn(100, 1) * 10000  # Preço com algum ruído

# Converter para um DataFrame do pandas
data = pd.DataFrame(data={'House Size (m^2)': house_size.flatten(), 'House Price ($)': house_price.flatten()})

# Dividir os dados em conjuntos de treino e teste
X = data[['House Size (m^2)']]
y = data['House Price ($)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo de regressão linear
model = LinearRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plotar os resultados
plt.scatter(X_test, y_test, color='blue', label='Dados Reais')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Previsão')
plt.xlabel('Tamanho da Casa (m^2)')
plt.ylabel('Preço da Casa ($)')
plt.title('Regressão Linear - Preço da Casa vs. Tamanho da Casa')
plt.legend()
plt.show()
