# common_utils.py
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')


def load_and_filter(path_csv: str, year: int = 2012) -> pd.DataFrame:
    """
    Đọc Reviews.csv, chuyển Time sang datetime và chỉ giữ đánh giá của năm `year`.
    """
    df = pd.read_csv('D:/Camera Roll/Users/ADMIN/Documents/GitHub/phantichdanhgiasanpham.github.io/phantichdanhgiasanpham/dataset/Reviews.csv')
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    df = df[df['Time'].dt.year == year]
    return df[['ProductId', 'Text', 'Score']]

def label_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gán nhãn sentiment từ Score thành positive/neutral/negative.
    """
    def score_to_label(score):
        if score >= 4:   return 'positive'
        elif score == 3: return 'neutral'
        else:            return 'negative'
    df['Label'] = df['Score'].apply(score_to_label)
    return df[['ProductId', 'Text', 'Label']]

def clean_text_series(text_series: pd.Series) -> pd.Series:
    """
    Tiền xử lý chuỗi: lowercase, bỏ ký tự đặc biệt, xóa stopwords, lemmatize.
    """
    stop_words = set(stopwords.words('english'))
    lemm = WordNetLemmatizer()
    def clean(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return " ".join(lemm.lemmatize(w) for w in text.split() if w not in stop_words)
    return text_series.apply(clean)


