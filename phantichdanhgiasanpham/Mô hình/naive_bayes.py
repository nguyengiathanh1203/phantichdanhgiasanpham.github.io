# naive_bayes.py
import pandas as pd
from common_utils import load_and_filter, label_sentiment, clean_text_series
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# 1. Đọc & tiền xử lý
df = load_and_filter('D:/Camera Roll/Users/ADMIN/Documents/GitHub/phantichdanhgiasanpham.github.io/phantichdanhgiasanpham/dataset/Reviews.csv')
df = label_sentiment(df)
df['Text'] = clean_text_series(df['Text'])

# 2. Tách X/y và vector hóa TF-IDF
X = df['Text']
y = df['Label']
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# 3. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42)

# 4. Huấn luyện Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)

# 5. Đánh giá
y_pred = nb.predict(X_test)

print("=== Naive Bayes Classification Report ===")

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%\n")

print(classification_report(y_test, y_pred))

# ===== THỐNG KÊ CẢM XÚC THEO SẢN PHẨM =====
print("\n=== Tóm tắt số lượng đánh giá và cảm xúc của 10 sản phẩm đầu tiên ===")
product_summary = df['ProductId'].value_counts().head(10)

for product_id, count in product_summary.items():
    print(f"\nProduct ID: {product_id} — Tổng số đánh giá: {count}")
    label_counts = df[df['ProductId'] == product_id]['Label'].value_counts()
    for label in ['positive', 'neutral', 'negative']:  # đảm bảo thứ tự hiển thị
        print(f"    {label}: {label_counts.get(label, 0)}")