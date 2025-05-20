# train_model.py
import pandas as pd
from common_utils import load_and_filter, label_sentiment, clean_text_series
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import joblib


df = load_and_filter('dataset/Reviews.csv')
df = label_sentiment(df)
df['Text'] = clean_text_series(df['Text'])




X = df['Text']
y = df['Label']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)




rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)





joblib.dump(rf, 'model_random_forest.pkl')
joblib.dump(vectorizer, 'vectorizer_tfidf.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print('Đã hoàn tất huấn luyện và lưu mô hình!')
