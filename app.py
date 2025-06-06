import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import streamlit as st

# Cấu hình trang
st.set_page_config(page_title="Phân loại khách hàng - Mall", layout="wide")

# Tiêu đề chính
st.title("Ứng dụng phân loại khách hàng tiềm năng từ dữ liệu Mall")

# Đọc dữ liệu
data = pd.read_csv("Mall_Customers.csv")
data.rename(columns={
    'CustomerID': 'Mã KH',
    'Genre': 'Giới tính',
    'Age': 'Tuổi',
    'Annual Income (k$)': 'Thu nhập (k$)',
    'Spending Score (1-100)': 'Điểm chi tiêu'
}, inplace=True)

# Phân cụm KMeans
X = data[['Tuổi', 'Thu nhập (k$)', 'Điểm chi tiêu']]
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
data['Cụm'] = kmeans.fit_predict(X)

# Lưu kết quả ra file CSV
output_file = "ket_qua_phan_cum.csv"
data.to_csv(output_file, index=False)

# Thông báo lưu file
st.success(f"Kết quả phân cụm đã được lưu vào file `{output_file}`")

# Bộ lọc dữ liệu
st.sidebar.header(" Bộ lọc dữ liệu")
genders = st.sidebar.multiselect("Giới tính", options=data['Giới tính'].unique(), default=data['Giới tính'].unique())
clusters = st.sidebar.multiselect("Cụm khách hàng", options=sorted(data['Cụm'].unique()), default=sorted(data['Cụm'].unique()))
age_range = st.sidebar.slider("Khoảng tuổi", int(data['Tuổi'].min()), int(data['Tuổi'].max()), (18, 70))
income_range = st.sidebar.slider("Thu nhập (k$)", int(data['Thu nhập (k$)'].min()), int(data['Thu nhập (k$)'].max()), (15, 135))

# Lọc dữ liệu
filtered_data = data[
    (data['Giới tính'].isin(genders)) &
    (data['Cụm'].isin(clusters)) &
    (data['Tuổi'].between(age_range[0], age_range[1])) &
    (data['Thu nhập (k$)'].between(income_range[0], income_range[1]))
]

# Hiển thị bảng dữ liệu
st.subheader(" Dữ liệu khách hàng đã phân cụm")
st.dataframe(filtered_data, use_container_width=True)

# Biểu đồ phân nhóm khách hàng (2D)
st.subheader(" Biểu đồ phân nhóm khách hàng theo thu nhập và điểm chi tiêu")
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    x='Thu nhập (k$)', y='Điểm chi tiêu', hue='Cụm', palette='Set2', data=filtered_data, ax=ax1
)
ax1.set_xlabel("Thu nhập (nghìn đô)")
ax1.set_ylabel("Điểm chi tiêu")
ax1.set_title("Phân nhóm khách hàng")
st.pyplot(fig1)

# Biểu đồ phụ: countplot theo cụm
st.subheader(" Số lượng khách hàng theo cụm")
fig2, ax2 = plt.subplots()
sns.countplot(x='Cụm', data=filtered_data, palette='Set2', ax=ax2)
ax2.set_title("Phân bố số lượng khách theo từng cụm")
st.pyplot(fig2)

# Histogram thu nhập
st.subheader(" Phân phối thu nhập của khách hàng")
fig3, ax3 = plt.subplots()
sns.histplot(filtered_data['Thu nhập (k$)'], kde=True, bins=20, color='skyblue', ax=ax3)
ax3.set_title("Phân phối thu nhập")
st.pyplot(fig3)

# Boxplot điểm chi tiêu theo cụm
st.subheader(" So sánh điểm chi tiêu giữa các cụm")
fig4, ax4 = plt.subplots()
sns.boxplot(x='Cụm', y='Điểm chi tiêu', data=filtered_data, palette='Set2', ax=ax4)
ax4.set_title("So sánh điểm chi tiêu giữa các cụm")
st.pyplot(fig4)

# Mô tả cụm
st.subheader(" Giải thích các cụm khách hàng (theo quan sát)")
st.markdown("""
- **Cụm 0**: Thu nhập thấp, điểm chi tiêu thấp → Khách hàng ít tiềm năng.
- **Cụm 1**: Thu nhập cao, điểm chi tiêu cao → Khách hàng VIP, rất tiềm năng.
- **Cụm 2**: Thu nhập trung bình, điểm chi tiêu cao → Khách hàng yêu thích mua sắm.
- **Cụm 3**: Thu nhập thấp, điểm chi tiêu cao → Khách hàng hay mua sắm, nhạy cảm giá.
- **Cụm 4**: Thu nhập cao, điểm chi tiêu thấp → Khách hàng giàu nhưng ít tiêu, cần chiến lược thúc đẩy mua sắm.
""")

# Nút tải kết quả
st.download_button(
    label="📥 Tải xuống kết quả phân cụm (.csv)",
    data=filtered_data.to_csv(index=False).encode('utf-8-sig'),
    file_name='ket_qua_phan_cum.csv',
    mime='text/csv'
)
