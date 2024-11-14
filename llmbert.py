import pandas as pd
import numpy as np
from transformers import BertForSequenceClassification, AutoTokenizer
from transformers import pipeline

# Importando arquivo do Web Scraping
df = pd.read_csv('app_reviews.csv')

# Refinando dados
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

# Salvando o arquivo CSV
df_short.to_csv('df_short.csv', index=False)