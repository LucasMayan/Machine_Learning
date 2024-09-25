from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Carregar o dataset MovieLens
data = Dataset.load_builtin('ml-100k')

# Dividir o dataset em treino e teste
trainset, testset = train_test_split(data, test_size=0.25)

# Criar o modelo SVD
algo = SVD()

# Treinar o modelo
algo.fit(trainset)

# Testar o modelo
predictions = algo.test(testset)

# Avaliar a precisão do modelo
accuracy.rmse(predictions)

# Fazer recomendações para um usuário específico
user_id = str(196)  # ID do usuário para o qual queremos fazer recomendações
item_ids = [str(i) for i in range(1, 1683)]  # IDs dos itens (filmes)

# Prever a nota para cada item (filme) para o usuário
predictions = [algo.predict(user_id, item_id) for item_id in item_ids]

# Ordenar os filmes pelas notas previstas
predictions.sort(key=lambda x: x.est, reverse=True)

# Exibir as 10 melhores recomendações
top_10_recommendations = predictions[:10]
for prediction in top_10_recommendations:
    print(f'Filme: {prediction.iid}, Nota prevista: {prediction.est}')
