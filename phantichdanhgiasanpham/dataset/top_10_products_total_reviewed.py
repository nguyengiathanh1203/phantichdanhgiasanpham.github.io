import pandas as pd

# Bước 1: Đọc dữ liệu từ file Reviews.csv
df = pd.read_csv('Reviews.csv')

# Bước 2: Đếm số lượng review theo từng ProductId
review_counts = df.groupby('ProductId').size().reset_index(name='ReviewCount')

# Bước 3: Sắp xếp giảm dần theo số lượng review và lấy top 10
top_10_reviews = review_counts.sort_values(by='ReviewCount', ascending=False).head(10)

# Bước 4: In kết quả ra màn hình
print("Top 10 sản phẩm có nhiều review nhất:")
print(top_10_reviews)
