import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import NMF

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Loading book dataset...")
df = pd.read_csv('book_dataset.csv')

print("Cleaning text data...")
df['clean_summary'] = df['summary'].apply(clean_text)
df = df.dropna(subset=['clean_summary', 'genres'])
df = df[df['clean_summary'].str.len() > 50]

print("Training Naive Bayes classifier...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(df['clean_summary'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['genres'].str.split(',').str[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Naive Bayes Accuracy: {accuracy:.3f}")
print(f"Naive Bayes F1-Score: {f1:.3f}")

print("Training Matrix Factorization...")
n_users = 200
n_books = len(df)

np.random.seed(42)
user_ratings = np.random.choice([0, 1, 2, 3, 4, 5], size=(n_users, n_books), p=[0.7, 0.05, 0.05, 0.05, 0.1, 0.05])

nmf_model = NMF(n_components=20, random_state=42)
user_features = nmf_model.fit_transform(user_ratings)
book_features = nmf_model.components_.T

reconstructed = np.dot(user_features, book_features.T)
mse = np.mean((user_ratings[user_ratings > 0] - reconstructed[user_ratings > 0]) ** 2)
rmse = np.sqrt(mse)

def precision_at_k(actual, predicted, k=5):
    predicted_k = predicted[:k]
    return len(set(predicted_k) & set(actual)) / k

precisions = []
for user_id in range(min(50, n_users)):
    user_actual = np.where(user_ratings[user_id] >= 4)[0]
    if len(user_actual) > 0:
        user_scores = reconstructed[user_id]
        user_predicted = np.argsort(user_scores)[::-1]
        precision = precision_at_k(user_actual, user_predicted, k=5)
        precisions.append(precision)

precision_at_5 = np.mean(precisions) if precisions else 0

print(f"Matrix Factorization RMSE: {rmse:.3f}")
print(f"Matrix Factorization Precision@5: {precision_at_5:.3f}")

print("\n" + "="*60)
print("BASELINE MODELS TRAINING RESULTS")
print("="*60)
print(f"{'Model':<25} {'Metric':<15} {'Score':<10}")
print("-"*60)
print(f"{'Naive Bayes':<25} {'Accuracy':<15} {accuracy:.3f}")
print(f"{'Naive Bayes':<25} {'F1-Score':<15} {f1:.3f}")
print(f"{'Matrix Factorization':<25} {'RMSE':<15} {rmse:.3f}")
print(f"{'Matrix Factorization':<25} {'Precision@5':<15} {precision_at_5:.3f}")


baseline_models = {
    'nb_model': nb_model,
    'vectorizer': vectorizer,
    'label_encoder': label_encoder,
    'nmf_model': nmf_model,
    'user_features': user_features,
    'book_features': book_features
}

with open('baseline_models.pkl', 'wb') as f:
    pickle.dump(baseline_models, f)

print("All models saved successfully!")