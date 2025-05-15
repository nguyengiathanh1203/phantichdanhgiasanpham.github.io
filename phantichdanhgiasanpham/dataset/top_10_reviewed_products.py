import pandas as pd

# Bước 1: Đọc dữ liệu
df = pd.read_csv('Reviews.csv')

# Bước 2: Đếm số lượng review cho mỗi ProductId
product_review_counts = df['ProductId'].value_counts()

# Bước 3: Lấy 10 ProductId có nhiều review nhất
top_10_product_ids = product_review_counts.head(10).index.tolist()

# Bước 4: Lọc toàn bộ các dòng liên quan đến 10 sản phẩm này
top_10_reviews = df[df['ProductId'].isin(top_10_product_ids)]

# Bước 5: Ghi kết quả vào file CSV
top_10_reviews.to_csv('top_10_reviewed_products.csv', index=False)

print("Đã lưu dữ liệu của 10 sản phẩm có nhiều review nhất vào 'top_10_products_full_reviews.csv'")
