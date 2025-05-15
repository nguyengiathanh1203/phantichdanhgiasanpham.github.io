import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Đọc dữ liệu
df = pd.read_csv('../dataset/top_10_reviewed_products.csv')

# 2. Tiền xử lý: chỉ dùng các review có điểm rõ ràng
df = df[df['Score'] != 3]
df['label'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)  # 1: tích cực, 0: tiêu cực

# 3. Làm sạch văn bản
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['Text'].apply(clean_text)

# 4. Tách dữ liệu
X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Vector hóa TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Huấn luyện các mô hình

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_preds = nb_model.predict(X_test_vec)

# # Logistic Regression
# lr_model = LogisticRegression(max_iter=1000)
# lr_model.fit(X_train_vec, y_train)
# lr_preds = lr_model.predict(X_test_vec)

# # Random Forest
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train_vec, y_train)
# rf_preds = rf_model.predict(X_test_vec)

# 7. Đánh giá mô hình
print("=== Naive Bayes ===")
print(classification_report(y_test, nb_preds))
print("Accuracy:", accuracy_score(y_test, nb_preds))

# print("\n=== Logistic Regression ===")
# print(classification_report(y_test, lr_preds))
# print("Accuracy:", accuracy_score(y_test, lr_preds))

# print("\n=== Random Forest ===")
# print(classification_report(y_test, rf_preds))
# print("Accuracy:", accuracy_score(y_test, rf_preds))
