import pandas as pd
from sklearn.metrics import accuracy_score
import pandas as pd
import streamlit as st

# Função para calcular porcentagens de sentimentos por banco
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

# Dicionário com os logotipos dos bancos
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

# Adicionando o arquivo com os sentimentos reais, sentimentos do LLM e as predições de cada um
df_short_new = pd.read_csv('df_short_new.csv')

# Refinando os dados
df_short_new['appId'] = df_short_new['appId'].replace(
    ['com.itau', 'com.nu.production', 'br.com.bb.android', 'com.bradesco', 'com.santander.app', 'com.c6bank.app', 'br.com.xp.carteira', 'br.com.intermedium'],
    ['Itau', 'Nubank', 'Banco do Brasil', 'Bradesco', 'Santander', 'C6 Bank', 'XP', 'Inter']
)

df_short_new['at'] = pd.to_datetime(df_short_new['at'])

# Criando a interface do Streamlit
st.title("Ranking de Apps de Bancos Mais Bem Avaliados da Google Play")

# Seleção de período com calendário
date_range = st.date_input("Selecione o Período para a Análise", [])

# Verifica se o usuário selecionou duas datas (início e fim do período)
if len(date_range) == 2:
    start_date, end_date = date_range
    df_reviews_filtered = df_short_new[(df_short_new['at'] >= pd.to_datetime(start_date)) & (df_short_new['at'] <= pd.to_datetime(end_date))]

    # Calculando os sentimentos por banco com a coluna 'sentiment'
    sentimentos_por_banco_sentiment = {}
    for index, row in df_reviews_filtered.iterrows():
        app_id = row['appId']
        sentiment = row['sentiment']

        if app_id not in sentimentos_por_banco_sentiment:
            sentimentos_por_banco_sentiment[app_id] = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0, 'TOTAL': 0}

        sentimentos_por_banco_sentiment[app_id][sentiment] += 1
        sentimentos_por_banco_sentiment[app_id]['TOTAL'] += 1

    # Calculando as porcentagens de sentimentos por banco com a coluna 'sentiment'
    porcentagens_por_banco_sentiment = calcular_porcentagens(sentimentos_por_banco_sentiment)

    # Calculando os sentimentos por banco com a coluna 'label'
    sentimentos_por_banco_label = {}
    for index, row in df_reviews_filtered.iterrows():
        app_id = row['appId']
        sentiment = row['label']

        if app_id not in sentimentos_por_banco_label:
            sentimentos_por_banco_label[app_id] = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0, 'TOTAL': 0}

        sentimentos_por_banco_label[app_id][sentiment] += 1
        sentimentos_por_banco_label[app_id]['TOTAL'] += 1

    # Calculando as porcentagens de sentimentos por banco com a coluna 'label'
    porcentagens_por_banco_label = calcular_porcentagens(sentimentos_por_banco_label)

    # Calculando os sentimentos previstos por banco com a coluna 'sentiment_pred'
    sentimentos_por_banco_sentiment_pred = {}
    for index, row in df_reviews_filtered.iterrows():
        app_id = row['appId']
        sentiment = row['sentiment_pred']  # Coluna de sentimentos previstos

        if app_id not in sentimentos_por_banco_sentiment_pred:
            sentimentos_por_banco_sentiment_pred[app_id] = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0, 'TOTAL': 0}

        sentimentos_por_banco_sentiment_pred[app_id][sentiment] += 1
        sentimentos_por_banco_sentiment_pred[app_id]['TOTAL'] += 1

    # Calculando as porcentagens de sentimentos previstos por banco
    porcentagens_por_banco_sentiment_pred = calcular_porcentagens(sentimentos_por_banco_sentiment_pred)

    # Calculando os sentimentos previstos por banco com a coluna 'label_pred'
    sentimentos_por_banco_label_pred = {}
    for index, row in df_reviews_filtered.iterrows():
        app_id = row['appId']
        sentiment = row['label_pred']  # Coluna de labels previstos

        if app_id not in sentimentos_por_banco_label_pred:
            sentimentos_por_banco_label_pred[app_id] = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0, 'TOTAL': 0}

        sentimentos_por_banco_label_pred[app_id][sentiment] += 1
        sentimentos_por_banco_label_pred[app_id]['TOTAL'] += 1

    # Calculando as porcentagens de sentimentos previstos por banco
    porcentagens_por_banco_label_pred = calcular_porcentagens(sentimentos_por_banco_label_pred)

    # Ordenando o ranking com base na porcentagem positiva (label)
    ranking_label = ranking_bancos(porcentagens_por_banco_label)

    # Exibindo o ranking
    st.subheader("Ranking - Comparativo Score Real vs Score LLM Bert")
    
    st.write("Banco - Score LLM Bert")
    st.write("Score Real - Predição Real - Predição LLM Bert - Diferença Real - Diferença LLM Bert")

    for i, (bank_id, percentages) in enumerate(ranking_label):
        sentiment_pos = porcentagens_por_banco_sentiment.get(bank_id, {}).get('POSITIVE', 0)
        label_pos = percentages['POSITIVE']
        sentiment_pred_pos = porcentagens_por_banco_sentiment_pred.get(bank_id, {}).get('POSITIVE', 0)
        label_pred_pos = porcentagens_por_banco_label_pred.get(bank_id, {}).get('POSITIVE', 0)

        # Calculando a diferença entre sentimento real e previsto
        diff_sentiment = sentiment_pos - sentiment_pred_pos  # Sentiment - Sentiment Pred
        diff_label = label_pos - label_pred_pos  # Label - Label Pred

        # Definindo o tamanho da imagem e do texto com base na posição no ranking
        if i == 0:
            image_size = 80
            text_size = "h3"
        elif i == 1:
            image_size = 70
            text_size = "h3"
        elif i == 2:
            image_size = 60
            text_size = "h3"
        else:
            image_size = 50
            text_size = "h3"

        # Exibindo as informações
        col1, col2 = st.columns([1, 5])
        
        with col1:
            st.image(bank_logos[bank_id], width=image_size)
        
        with col2:
            st.markdown(f"""
                <div style='font-size: 24px;'>
                    <strong>{i + 1}. {bank_id}: {label_pos:.2f}%</strong><br>
                    <span style='font-size: 18px;'>{sentiment_pos:.2f}% - {sentiment_pred_pos:.2f}% - {label_pred_pos:.2f}% - {diff_sentiment:.2f}% - {diff_label:.2f}%</span>
                </div>
            """, unsafe_allow_html=True)
            
# Calculando a acurácia
accuracy = accuracy_score(df_short_new['sentiment'], df_short_new['label'])
accuracy_percent = accuracy * 100
st.markdown(f"<h3 style='font-size:14px;'>Acurácia LLM Bert: {accuracy_percent:.2f}%</h3>", unsafe_allow_html=True)