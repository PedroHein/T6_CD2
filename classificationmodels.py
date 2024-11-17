import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Carregando o arquivo
df_short = pd.read_csv('df_short.csv')

# Vetorizando os comentários
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df_short['content']).toarray()

# Dividindo o conjunto de dados em treino e teste - Score Real
def train_and_evaluate_sentiment(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "RandomForestClassifier": RandomForestClassifier(),
        "BernoulliNB": BernoulliNB(),
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB()
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Score Real - {model_name} Acuracidade: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

    print(f"\nMelhor modelo é o {best_model_name} com acuracidade de {best_accuracy:.4f}")
    return best_model

# Dividindo o conjunto de dados em treino e teste - Score LLM Bert
def train_and_evaluate_label(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "RandomForestClassifier": RandomForestClassifier(),
        "BernoulliNB": BernoulliNB(),
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB()
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Score LLM Bert - {model_name} Acuracidade: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

    print(f"\nMelhor modelo é o {best_model_name} com acuracidade de {best_accuracy:.4f}")
    return best_model

# Aplicando para 'sentiment' - Score Real
print("Treino do Score Real:")
y_sentiment = df_short['sentiment']
best_model_sentiment = train_and_evaluate_sentiment(X, y_sentiment)

# Aplicando para 'label' - Score LLM Bert
print("\nTreino do Score LLM Bert:")
y_label = df_short['label']
best_model_label = train_and_evaluate_label(X, y_label)

# Salvando no df
df_short['sentiment_pred'] = best_model_sentiment.predict(X)
df_short['label_pred'] = best_model_label.predict(X)

df_short_new = df_short[['appId', 'content', 'sentiment', 'sentiment_pred', 'label', 'label_pred', 'at']]
df_short_new.to_csv('df_short_new.csv', index=False)
