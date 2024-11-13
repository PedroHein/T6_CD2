# import torch
import pandas as pd
# from transformers import AutoTokenizer, BertForSequenceClassification, pipeline
import streamlit as st
import pandas as pd
import numpy as np
# from tqdm import tqdm
from google_play_scraper import Sort, reviews
# import os
from sklearn.metrics import accuracy_score, classification_report


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

# # Web scraping para obter as avaliações dos aplicativos e salvar no CSV
# def obter_reviews():
#     app_reviews = []
#     for ap in tqdm(apps_ids):
#         rvs, _ = reviews(
#             ap,
#             lang='pt',
#             country='br',
#             sort=Sort.NEWEST,
#             count=8000,
#         )
#         for r in rvs:
#             r['sortOrder'] = 'newest'
#             r['appId'] = ap
#         app_reviews.extend(rvs)
    
#     # Convertendo as avaliações para DataFrame
#     df_reviews = pd.DataFrame(app_reviews)
    
#     # Salvando as avaliações em um arquivo CSV
#     df_reviews.to_csv('app_reviews.csv', index=False)
#     return df_reviews

# # Carregar os dados de avaliações (se já existir) ou executar o scraping
# if os.path.exists('app_reviews.csv'):
#     df_reviews = pd.read_csv('app_reviews.csv')
# else:
#     df_reviews = obter_reviews()

# Função para calcular porcentagens de sentimentos
def calcular_porcentagens(sentimentos_por_banco):
    porcentagens_por_banco = {}
    for bank_id, sentiments in sentimentos_por_banco.items():
        total_reviews = sentiments['TOTAL']
        if total_reviews > 0:
            porcentagens_por_banco[bank_id] = {
                'POSITIVE': (sentiments['POSITIVE'] / total_reviews) * 100,
                'NEGATIVE': (sentiments['NEGATIVE'] / total_reviews) * 100,
                'NEUTRAL': (sentiments['NEUTRAL'] / total_reviews) * 100
            }
    return porcentagens_por_banco

# Função para classificar os bancos com base na porcentagem de sentimento positivo
def ranking_bancos(sentiment_percentages):
    ranked_bancos = sorted(sentiment_percentages.items(), key=lambda item: item[1]['POSITIVE'], reverse=True)
    return ranked_bancos

# Dicionário para mapear os logotipos dos bancos
bank_logos = {
    "Itau": "images/itau-logo.png",
    "Nubank": "images/nubank-logo.png",
    "Banco do Brasil": "images/banco-do-brasil.png",
    "Bradesco": "images/bradesco.png",
    "Santander": "images/santander-br.png",
    "C6 Bank": "images/c6bank-logo.png",
    "XP": "images/xp-logo.png",
    "Inter": "images/banco-inter.png"
}

# df_reviews['sentiment'] = df_reviews.score.replace([1,2,3,4,5],['NEGATIVE','NEUTRAL','NEUTRAL','NEUTRAL','POSITIVE'])
# df_reviews = df_reviews[['appId','content','score','sentiment', 'at']]

# df_reviews = pd.read_csv('app_reviewspequeno.csv')

# # LLM Bert - FinBERT
# np.random.seed(42)
# df_short = df_reviews.iloc[ np.random.randint(0, len(df_reviews), size=df_reviews.shape[0]) ].reset_index(drop=True)
# df_short.rename(columns={'score':'score_app'}, inplace=True)

# # Carregar o tokenizador e o modelo FinBERT-PT-BR
# tokenizer = AutoTokenizer.from_pretrained("lucas-leme/FinBERT-PT-BR")
# model = BertForSequenceClassification.from_pretrained("lucas-leme/FinBERT-PT-BR")

# # Criar a pipeline para classificação de texto
# pipeline = pipeline(task='text-classification', model=model, tokenizer=tokenizer)
# print(pipeline(['O aplicativo é muito bom', 'O aplicativo é muito ruim']))

# results = pipeline(df_short['content'].tolist())

# df_short = pd.concat([df_short, pd.DataFrame(results)], axis=1)

df_short = pd.read_csv('df_short.csv')

# Refinando os dados
df_short['appId'] = df_short['appId'].replace(
    ['com.itau', 'com.nu.production', 'br.com.bb.android', 'com.bradesco', 'com.santander.app', 'com.c6bank.app', 'br.com.xp.carteira', 'br.com.intermedium'],
    ['Itau', 'Nubank', 'Banco do Brasil', 'Bradesco', 'Santander', 'C6 Bank', 'XP', 'Inter']
)

df_short['at'] = pd.to_datetime(df_short['at'])

# Streamlit
st.title("Ranking de Apps de Bancos Mais Bem Avaliados da Google Play")

# Seleção de período com calendário
date_range = st.date_input("Selecione o Período para a Análise", [])

# Verifica se o usuário selecionou duas datas (início e fim do período)
if len(date_range) == 2:
    start_date, end_date = date_range
    df_reviews_filtered = df_short[(df_short['at'] >= pd.to_datetime(start_date)) & (df_short['at'] <= pd.to_datetime(end_date))]

    # Calculando os sentimentos por banco
    sentimentos_por_banco = {}
    for index, row in df_reviews_filtered.iterrows():
        app_id = row['appId']
        sentiment = row['label']

        if app_id not in sentimentos_por_banco:
            sentimentos_por_banco[app_id] = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0, 'TOTAL': 0}

        sentimentos_por_banco[app_id][sentiment] += 1
        sentimentos_por_banco[app_id]['TOTAL'] += 1

    # Calculando as porcentagens de sentimentos por banco
    porcentagens_por_banco = calcular_porcentagens(sentimentos_por_banco)

    # Ordenando o ranking com base na porcentagem positiva
    ranking = ranking_bancos(porcentagens_por_banco)

    st.subheader(f"Ranking Positivo do Período Selecionado")

    for i, (bank_id, percentages) in enumerate(ranking):
        col1, col2 = st.columns([1, 5])
        
        with col1:
            if i == 0:
                st.image(bank_logos[bank_id], width=80)
            elif i == 1:
                st.image(bank_logos[bank_id], width=70)
            elif i == 2:
                st.image(bank_logos[bank_id], width=60)
            else:
                st.image(bank_logos[bank_id], width=50)

        with col2:
            if i == 0:
                st.markdown(f"<h1 style='color: white; display: flex; align-items: center; height: 80px;'>{i + 1}. {bank_id}: {percentages['POSITIVE']:.2f}%</h1>", unsafe_allow_html=True)
            elif i == 1:
                st.markdown(f"<h2 style='color: white; display: flex; align-items: center; height: 70px;'>{i + 1}. {bank_id}: {percentages['POSITIVE']:.2f}%</h2>", unsafe_allow_html=True)
            elif i == 2:
                st.markdown(f"<h3 style='color: white; display: flex; align-items: center; height: 60px;'>{i + 1}. {bank_id}: {percentages['POSITIVE']:.2f}%</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color: white; display: flex; align-items: center; height: 50px;'>{i + 1}. {bank_id}: {percentages['POSITIVE']:.2f}%</p>", unsafe_allow_html=True)

# Calculando a acurácia
accuracy = accuracy_score(df_short['sentiment'], df_short['label'])
accuracy_percent = accuracy * 100
st.markdown(f"<h3 style='font-size:14px;'>Acurácia LLM Bert: {accuracy_percent:.2f}%</h3>", unsafe_allow_html=True)

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Carregar o DataFrame (já preparado previamente)
df_short = pd.read_csv('df_short.csv')

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(df_short['content'], df_short['sentiment'], test_size=0.2, random_state=42)

# Vetorização com TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Função para exibir métricas
def exibir_metricas(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    st.write(f"### {model_name} Metrics:")
    st.markdown(f"**Accuracy:** {accuracy:.2f}")
    st.write(pd.DataFrame(report).transpose())
    st.markdown("")

# Treinar e avaliar os modelos
models = {
    'Logistic Regression': LogisticRegression(),
    'KNeighbors Classifier': KNeighborsClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Bernoulli Naive Bayes': BernoulliNB(),
    'Gaussian Naive Bayes': GaussianNB(),
    'Multinomial Naive Bayes': MultinomialNB(),
}

for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    exibir_metricas(model_name, y_test, y_pred)

# Adicionando a acurácia do FinBERT ou qualquer outro modelo que você deseje
# Aqui você já tem o cálculo da acurácia do modelo Bert implementado no código
accuracy = accuracy_score(df_short['sentiment'], df_short['label'])
accuracy_percent = accuracy * 100
st.markdown(f"<h3 style='font-size:14px;'>Acurácia LLM Bert: {accuracy_percent:.2f}%</h3>", unsafe_allow_html=True)
