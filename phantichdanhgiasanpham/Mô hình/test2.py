import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc file CSV
df = pd.read_csv("../dataset/Reviews.csv")

# Chuyển cột Time (Unix timestamp) sang datetime
df['Time'] = pd.to_datetime(df['Time'], unit='s')

# Lọc dữ liệu chỉ lấy năm 2012
df_2012 = df[df['Time'].dt.year == 2012]

def label_sentiment(score):
    if score in [1, 2]:
        return 'negative'
    elif score == 3:
        return 'neutral'
    else:
        return 'positive'

df_2012['Sentiment'] = df_2012['Score'].apply(label_sentiment)

# Xóa dòng bị thiếu (nếu có)
df_2012 = df_2012.dropna(subset=['Text'])

# Lấy text và label
texts = df_2012['Text']
labels = df_2012['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_preds = nb_model.predict(X_test_tfidf)

nb_accuracy = accuracy_score(y_test, nb_preds)

print("🔹 Naive Bayes Report:")
print(classification_report(y_test, nb_preds))
print("🔹 Naive Bayes Accuracy:", round(nb_accuracy * 100, 2), "%")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_preds = lr_model.predict(X_test_tfidf)

lr_accuracy = accuracy_score(y_test, lr_preds)

print("🔹 Logistic Regression Report:")
print(classification_report(y_test, lr_preds))
print("🔹 Logistic Regression Accuracy:", round(lr_accuracy * 100, 2), "%")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
rf_preds = rf_model.predict(X_test_tfidf)

rf_accuracy = accuracy_score(y_test, rf_preds)

print("🔹 Random Forest Report:")
print(classification_report(y_test, rf_preds))
print("🔹 Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%")