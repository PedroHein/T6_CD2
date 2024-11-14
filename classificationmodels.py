import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Dividindo o conjunto de dados para previsão de 'sentiment'
X_train_sentiment, X_test_sentiment, y_train_sentiment, y_test_sentiment = train_test_split(X, y_sentiment, test_size=0.2, random_state=42)

# Dividindo o conjunto de dados para previsão de 'label'
X_train_label, X_test_label, y_train_label, y_test_label = train_test_split(X, y_label, test_size=0.2, random_state=42)

# Modelos que iremos explorar
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "BernoulliNB": BernoulliNB(),
    "GaussianNB": GaussianNB(),
    "MultinomialNB": MultinomialNB()
}

# Função para treinar e avaliar os modelos
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
    
    # Retornando o melhor modelo e sua respectica acuracia
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

# Decodificando para os valores originais
df_short['sentiment_pred'] = le_sentiment.inverse_transform(df_short['sentiment_pred'])
df_short['label_pred'] = le_label.inverse_transform(df_short['label_pred'])

# Decodificando para valores originais
df_short['sentiment'] = le_sentiment.inverse_transform(df_short['sentiment'])
df_short['label'] = le_label.inverse_transform(df_short['label'])

# DataFrame final
df_short_new = df_short[['appId','content', 'sentiment', 'sentiment_pred', 'label', 'label_pred', 'at']]

# Gerando o arquivo final 
df_short_new.to_csv('df_short_new.csv', index=False)