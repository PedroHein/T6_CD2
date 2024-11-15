# Ranking de Apps de Bancos Mais Bem Avaliados da Google Play
Trabalho semestral de Ciência de Dados do Instituto Mauá de Tecnologia
- Fernando Henriques Neto – 18.00931-0
- Guilherme Sanches Rossi – 19.02404-5
- Pedro Henrique Sant Anna Hein – 20.00134-7
- Matheus Coelho Rocha Pinto – 20.00391-9

## Sobre o projeto
Este estudo desenvolve um sistema automatizado para medir a satisfação dos usuários de aplicativos bancários, utilizando o LLM Bert e aprendizado de máquina para análise de sentimentos. Com dados coletados da Google Play Store, o sistema cria um ranking dos aplicativos mais bem avaliados em um intervalo de datas pré-definidas, permitindo uma visão quantitativa e qualitativa sobre a experiência dos clientes. A interface em Streamlit e a execução em Docker facilitam a visualização e acessibilidade dos dados para gestores, que podem acompanhar a evolução semanal das percepções dos usuários. Este sistema pode orientar melhorias contínuas, ajudando instituições financeiras a adaptarem-se melhor às demandas do mercado digital.

## Documentação do Projeto
O Projeto é composto de 2 arquivos principais:
- main.py
- webscraping.py
- llmbert.py
- classificationmodels.py

Apenas o arquivo main.py deve ser executado. Os outros arquivos foram utilizados para gerar nossa base final com os sentimentos de cada modelo.

### webscraping.py
Neste arquivo utilizamos a biblioteca google_play_scraper para extrair os comentários de cada aplicativo do Google Play Store. A primeira etapa consistiu na definição dos aplicativos a serem analisados, foram eles: 
- Itaú
- Nubank
- Banco do Brasil
- Bradesco
- Santander
- C6 bank
- XP investimentos
- Banco Inter

Posteriormente, aplicamos alguns filtros de idioma, aplicamos também o sort.newest para obter as avaliações mais recentes, e por fim limitamos um número máximo de 8000 avaliações por aplicativo, o que nos proporcionou uma amostra ampla e variada para a análise de sentimentos.
Após rodar o código, extraimos o app_reviews.csv.

### llmbert.py
Com o arquivo do WebScraping gerado, utilizamos o FinBERT (modelo aberto https://huggingface.co/lucas-leme/FinBERT-PT-BR treinado com 1.4 milhões de textos do mercado financeiro em português para análise de sentimento) para realizar as análises de sentimentos e classificações das avaliações de cada usuário. Ele nos permite classificar se a avaliação foi positiva, neutra ou negativa de acordo com o comentário do próprio usuário. 
Após executar o código, extraimos o arquivo df_short.csv.

### classificationmodels.py
Com o arquivo do LLM Bert gerado pudemos aplicar o pré-processamento, treino e avaliação de modelos de classificação. Primeiro, utilizamos o LabelEncoder para converter as colunas de sentimentos e rótulos em valores numéricos. Em seguida, vetorizamos os textos dos comentários com TfidfVectorizer. Dividimos o dataset para treinar e testar os modelos, então treinamos e comparamos diversos modelos de classificação, escolhendo o RandomForestClassifier como o melhor para prever os sentimentos e rótulos (Acurácia de aproximadamente 86%). Após gerar as previsões salvamos o resultado final em df_short_new.csv.

### main.py
Neste código, nós criamos uma interface no Streamlit para analisar e comparar as avaliações ue até então analisamos. Primeiramente, carregamos e refinamos os dados, em seguida, calculamos as porcentagens de sentimentos (positivo, negativo e neutro) por aplicativo, utilizando as colunas de sentimentos reais e previstas pelos modelos (sentiment, label, sentiment_pred e label_pred). Seguindo, ordenamos os bancos com base na porcentagem de sentimentos positivos e exibimos um ranking que compara os valores reais e previstos de cada banco.

## Conclusões
### Video de apresentação

https://github.com/user-attachments/assets/1d780c03-b9bd-4678-871b-2b9e1bae6510

[Acesse o vídeo aqui!](https://youtu.be/f_4I5dI4NwM)

### Artigo
[Acesse o artigo aqui!](Artigo%20Projeto%20Semestral%20-%20Ranking%20de%20Sentimentos.pdf)

## Como executar o código
