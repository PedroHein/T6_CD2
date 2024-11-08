import streamlit as st
import pandas as pd

# Função para calcular porcentagens de sentimentos
def calcular_porcentagens(sentimentos_por_banco):
    porcentagens_por_banco = {}
    for bank_id, sentiments in sentimentos_por_banco.items():
        total_reviews = sentiments['TOTAL']
        if total_reviews > 0:
            porcentagens_por_banco[bank_id] = {
                'POSITIVO': (sentiments['POSITIVO'] / total_reviews) * 100,
                'NEGATIVO': (sentiments['NEGATIVO'] / total_reviews) * 100,
                'NEUTRO': (sentiments['NEUTRO'] / total_reviews) * 100
            }
    return porcentagens_por_banco

# Função para classificar os bancos com base na porcentagem de sentimento positivo
def ranking_bancos(sentiment_percentages):
    ranked_bancos = sorted(sentiment_percentages.items(), key=lambda item: item[1]['POSITIVO'], reverse=True)
    return ranked_bancos

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

# Carregamento do arquivo de reviews
df_reviews = pd.read_csv('app_reviews.csv')

# Refinando os dados
df_reviews['sentiment'] = df_reviews.score.replace([1, 2, 3, 4, 5], ['NEGATIVO', 'NEUTRO', 'NEUTRO', 'NEUTRO', 'POSITIVO'])
df_reviews = df_reviews[['appId', 'content', 'score', 'sentiment', 'at']]
df_reviews['appId'] = df_reviews['appId'].replace(
    ['com.itau', 'com.nu.production', 'br.com.bb.android', 'com.bradesco', 'com.santander.app', 'com.c6bank.app', 'br.com.xp.carteira', 'br.com.intermedium'],
    ['Itau', 'Nubank', 'Banco do Brasil', 'Bradesco', 'Santander', 'C6 Bank', 'XP', 'Inter']
)
df_reviews['at'] = pd.to_datetime(df_reviews['at']).dt.strftime('%Y-%m-%d %H:%M:%S')
df_reviews['at'] = df_reviews['at'].str[:-9]
df_reviews['at'] = pd.to_datetime(df_reviews['at'])

# Streamlit
st.title("Ranking Semanal de Apps de Bancos Mais Bem Avaliados da Google Play")

# Seleção de período com calendário
date_range = st.date_input("Selecione o Período para a Análise", [])

# Verifica se o usuário selecionou duas datas (início e fim do período)
if len(date_range) == 2:
    start_date, end_date = date_range
    df_reviews_filtered = df_reviews[(df_reviews['at'] >= pd.to_datetime(start_date)) & (df_reviews['at'] <= pd.to_datetime(end_date))]

    # Calculando os sentimentos por banco
    sentimentos_por_banco = {}
    for index, row in df_reviews_filtered.iterrows():
        app_id = row['appId']
        sentiment = row['sentiment']

        if app_id not in sentimentos_por_banco:
            sentimentos_por_banco[app_id] = {'POSITIVO': 0, 'NEGATIVO': 0, 'NEUTRO': 0, 'TOTAL': 0}

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
                st.markdown(f"<h1 style='color: white; display: flex; align-items: center; height: 80px;'>{i + 1}. {bank_id}: {percentages['POSITIVO']:.2f}%</h1>", unsafe_allow_html=True)
            elif i == 1:
                st.markdown(f"<h2 style='color: white; display: flex; align-items: center; height: 70px;'>{i + 1}. {bank_id}: {percentages['POSITIVO']:.2f}%</h2>", unsafe_allow_html=True)
            elif i == 2:
                st.markdown(f"<h3 style='color: white; display: flex; align-items: center; height: 60px;'>{i + 1}. {bank_id}: {percentages['POSITIVO']:.2f}%</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color: white; display: flex; align-items: center; height: 50px;'>{i + 1}. {bank_id}: {percentages['POSITIVO']:.2f}%</p>", unsafe_allow_html=True)