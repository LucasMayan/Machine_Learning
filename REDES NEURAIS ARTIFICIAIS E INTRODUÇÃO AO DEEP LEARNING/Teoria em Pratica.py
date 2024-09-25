import numpy as np

# Função de ativação - Função sinal
def sign_activation(x):
    return np.where(x > 0, 1, -1)

# Dados de entrada para os Quadros 1 e 2
X_quadro1 = np.array([[1, 0, 1],
                      [0, 1, 1],
                      [0, 0, 1]])

X_quadro2 = np.array([[1, 1, 1],
                      [1, 0, 1],
                      [0, 0, 1]])

# Saídas desejadas para os Quadros 1 e 2
y_quadro1 = np.array([1, -1, -1])
y_quadro2 = np.array([1, 1, -1])

# Inicialização dos pesos aleatórios
np.random.seed(0)
W = np.random.randn(X_quadro1.shape[1])

# Taxa de aprendizagem
eta = 1 / (X_quadro1.shape[0] + X_quadro1.shape[1])

# Loop de treinamento para Quadro 1
epochs = 1000
for epoch in range(epochs):
    # Calcular a saída predita
    y_pred = sign_activation(np.dot(X_quadro1, W))
    
    # Calcular o erro
    error = y_quadro1 - y_pred
    
    # Atualizar os pesos usando a regra de atualização
    delta_W = eta * np.dot(error, X_quadro1)
    W += delta_W

# Avaliar o modelo treinado nos dados do Quadro 2
y_pred_quadro2 = sign_activation(np.dot(X_quadro2, W))
print("Saída predita para o Quadro 2:", y_pred_quadro2)
