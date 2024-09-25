# pip install textblob
# pip install spacy
import spacy

# Carregar o modelo de linguagem em português do spaCy
nlp = spacy.load("pt_core_news_sm")

# Texto de exemplo
texto = "João e Maria foram ao Parque Ibirapuera em São Paulo no último sábado. Eles ficaram encantados com a beleza do lugar e decidiram visitar o Museu de Arte Moderna, que fica dentro do parque. À noite, jantaram no Restaurante Fasano, onde tiveram uma experiência gastronômica incrível."

# Processar o texto
doc = nlp(texto)

# Extração de entidades nomeadas
print("Entidades Nomeadas:")
for ent in doc.ents:
    print(ent.text, ent.label_)

# Análise de sentimentos (usando um exemplo simplificado de polaridade)
from textblob import TextBlob

# Função para traduzir texto para inglês, já que TextBlob suporta melhor o inglês para análise de sentimentos
from googletrans import Translator
translator = Translator()
traducao = translator.translate(texto, src='pt', dest='en')
texto_em_ingles = traducao.text

# Analisar o sentimento
blob = TextBlob(texto_em_ingles)
sentimento = blob.sentiment

print("\nAnálise de Sentimentos:")
print(f"Polaridade: {sentimento.polarity}, Subjetividade: {sentimento.subjectivity}")
