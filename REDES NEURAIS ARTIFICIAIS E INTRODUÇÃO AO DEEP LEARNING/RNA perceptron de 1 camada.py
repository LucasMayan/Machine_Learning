import numpy as np
class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(num_features + 1)  # Inclui o bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        # Adiciona o bias (valor 1) à entrada
        X = np.insert(X, 0, 1, axis=1)
        return self.activation_function(np.dot(X, self.weights))

    def fit(self, X, y):
        # Adiciona o bias (valor 1) à entrada
        X = np.insert(X, 0, 1, axis=1)
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = self.activation_function(np.dot(xi, self.weights))
                update = self.learning_rate * (target - prediction)
                self.weights += update * xi
# Dados de entrada (X) e saída (y)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Saída desejada (AND lógico)
y = np.array([0, 0, 0, 1])

# Inicializa o perceptron
perceptron = Perceptron(num_features=2, learning_rate=0.1, epochs=10)

# Treina o perceptron
perceptron.fit(X, y)

# Faz previsões
predictions = perceptron.predict(X)
print("Previsões:", predictions)
