import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from tqdm import tqdm
from google_play_scraper import Sort, reviews
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Google play scraper: https://github.com/JoMingyu/google-play-scraper
# !pip install google_play_scraper

# Aplicativos a serem analisados
apps_ids = [
    'com.itau', 
    'com.nu.production', 
    'br.com.bb.android', 
    'com.bradesco',
    'com.santander.app', 
    'com.c6bank.app', 
    'br.com.xp.carteira', 
    'br.com.intermedium'
]

# Web scraping para obter as avaliações dos aplicativos e salvar no CSV
def obter_reviews():
    app_reviews = []
    for ap in tqdm(apps_ids):
        rvs, _ = reviews(
            ap,
            lang='pt',
            country='br',
            sort=Sort.NEWEST,
            count=8000,
        )
        for r in rvs:
            r['sortOrder'] = 'newest'
            r['appId'] = ap
        app_reviews.extend(rvs)
    
    # Convertendo as avaliações para DataFrame
    df_reviews = pd.DataFrame(app_reviews)
    
    # Salvando as avaliações em um arquivo CSV
    df_reviews.to_csv('app_reviews.csv', index=False)
    return df_reviews

#Importando aruivo do Web Scraping
df = pd.read_csv('app_reviews.csv')

#Refinando dados
df['sentiment'] = df.score.replace([1,2,3,4,5],['NEGATIVE','NEUTRAL','NEUTRAL','NEUTRAL','POSITIVE'])
df = df[['appId','content','score','sentiment', 'at']]

# Aplicando o LLM BERT - FinBERT
np.random.seed(42)
df_short = df.iloc[ np.random.randint(0, len(df), size=100) ].reset_index(drop=True)
df_short.rename(columns={'score':'score_app'}, inplace=True)

tokenizer = AutoTokenizer.from_pretrained("lucas-leme/FinBERT-PT-BR")
model = BertForSequenceClassification.from_pretrained("lucas-leme/FinBERT-PT-BR")

pipeline = pipeline(task='text-classification', model=model, tokenizer=tokenizer)
print(pipeline(['Aplicativo muito bom e prático', 'O aplicaivo é muito ruim']))

results = pipeline(df_short['content'].tolist())

df_short = pd.concat([df_short, pd.DataFrame(results)], axis=1)
df_short.to_csv('df_short.csv', index=False)


# Aplicando o LabelEnconder e os modelos de classificação
# Com o arquivo df_short em mãos começamos as previsões
df_short = pd.read_csv('df_short.csv')

# LabelEncoder
le_sentiment = LabelEncoder()
le_label = LabelEncoder()
df_short['sentiment'] = le_sentiment.fit_transform(df_short['sentiment'])
df_short['label'] = le_label.fit_transform(df_short['label'])

# Vetorizando os comentários
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df_short['content']).toarray()

# Separando variáveis dependentes para os dois modelos
y_sentiment = df_short['sentiment']
y_label = df_short['label']

# Dividir o conjunto de dados para previsão de 'sentiment'
X_train_sentiment, X_test_sentiment, y_train_sentiment, y_test_sentiment = train_test_split(X, y_sentiment, test_size=0.2, random_state=42)

# Dividir o conjunto de dados para previsão de 'label'
X_train_label, X_test_label, y_train_label, y_test_label = train_test_split(X, y_label, test_size=0.2, random_state=42)

# Dicionário dos modelos
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "BernoulliNB": BernoulliNB(),
    "GaussianNB": GaussianNB(),
    "MultinomialNB": MultinomialNB()
}

# Função para treinar e avaliar modelos, retornando o melhor modelo e sua acurácia
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    for model_name, model in models.items():
        # Treina cada modelo
        if model_name == "GaussianNB":
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        print(f"{model_name} Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name
    
    print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    return best_model

# Descobrimos que o melhor modelo para ambas as colunas é o RandomForesClassifier

# Treinando e selecionando o melhor modelo para 'sentiment'
print("Sentiment Prediction:")
best_model_sentiment = train_and_evaluate(models, X_train_sentiment, X_test_sentiment, y_train_sentiment, y_test_sentiment)

# Treinando e selecionando o melhor modelo para 'label'
print("\nLabel Prediction:")
best_model_label = train_and_evaluate(models, X_train_label, X_test_label, y_train_label, y_test_label)

# Previsões completas e salvando no DataFrame original
df_short['sentiment_pred'] = best_model_sentiment.predict(X)
df_short['label_pred'] = best_model_label.predict(X)

# Decodificando para valores originais
df_short['sentiment_pred'] = le_sentiment.inverse_transform(df_short['sentiment_pred'])
df_short['label_pred'] = le_label.inverse_transform(df_short['label_pred'])

# Exibindo o DataFrame final com as previsões
df_short[['appId', 'content', 'sentiment', 'sentiment_pred', 'label', 'label_pred', 'at']]

# Decodificando para valores originais
df_short['sentiment'] = le_sentiment.inverse_transform(df_short['sentiment'])
df_short['label'] = le_label.inverse_transform(df_short['label'])

# DataFrame final
df_short_new = df_short[['appId','content', 'sentiment', 'sentiment_pred', 'label', 'label_pred', 'at']]

# Gerando o arquivo final 
df_short_new.to_csv('df_short_new.csv', index=False)