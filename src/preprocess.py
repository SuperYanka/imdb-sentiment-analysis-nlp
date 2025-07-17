import pandas as pd
import numpy as np

import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

stop_words = set(stopwords.words('english'))

def cleaned_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Удаляем HTML-теги
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Удаляем все, кроме букв и пробелов
   
    words = text.split()
    filtered = [word for word in words if word not in stop_words]

    return ' '.join(filtered) 


def vectorize_review(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    print("Файл загружен. Колонки:", df.columns)

    print("Пробуем применить функцию clean_text...")
    df['clean_text'] = df['review'].apply(cleaned_text)
    print("Колонка clean_text успешно создана")
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    X = df['clean_text']
    y = df['label']   

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer()
    X_train_vec  = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test, tfidf
  

def load_and_preprocess_text(filepath):
    df = pd.read_csv(filepath)

    df['tokens'] = df['review'].apply(cleaned_text)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    X = df['tokens']
    y = df['label']

    X_train_tokens, X_test_tokens, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучаем Word2Vec только на тренировочных данных
    w2v_model = Word2Vec(
        sentences=X_train_tokens,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4
    )

    # Превращаем токены в усреднённые векторы
    X_train_vec = np.array([vectorize_review(tokens, w2v_model) for tokens in X_train_tokens])
    X_test_vec = np.array([vectorize_review(tokens, w2v_model) for tokens in X_test_tokens])

    return X_train_vec, X_test_vec, y_train, y_test, w2v_model
   



