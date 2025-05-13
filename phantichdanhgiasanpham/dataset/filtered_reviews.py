import pandas as pd

# Bước 1: Đọc dữ liệu từ file Reviews.csv
df = pd.read_csv('Reviews.csv')

# Bước 2: Lọc các dòng có ProductId là 'B002QWP89S'
filtered_df = df[df['ProductId'] == 'B002QWP89S']

# Bước 3: Xuất dữ liệu đã lọc ra file CSV mới
filtered_df.to_csv('filtered_reviews_B002QWP89S.csv', index=False)

print("Đã lưu dữ liệu đã lọc vào file 'filtered_reviews_B002QWP89S.csv'")