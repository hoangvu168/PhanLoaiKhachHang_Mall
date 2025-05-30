import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Bước 1: Đọc dữ liệu
data = pd.read_csv("Mall_Customers.csv")
print(data.head())

# Bước 2: Lấy các đặc trưng để phân loại
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Bước 3: Tìm số cụm tối ưu (Elbow method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Phương pháp Elbow để chọn số cụm")
plt.xlabel("Số cụm")
plt.ylabel("WCSS")
plt.show()

# Bước 4: Phân nhóm với KMeans (giả sử chọn 5 cụm)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Thêm cột nhóm vào DataFrame
data['Cluster'] = y_kmeans

# Bước 5: Vẽ biểu đồ phân cụm
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set1',
    data=data
)
plt.title("Phân nhóm khách hàng theo thu nhập và điểm chi tiêu")
plt.show()
