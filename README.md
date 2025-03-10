
# 🛒 Online Retail Dataset - Machine Learning Project (Unsupervised Learning)

## 📌 Overview
โปรเจกต์นี้เป็นการวิเคราะห์และพัฒนาโมเดล Machine Learning โดยใช้ **Online Retail Dataset** จาก Kaggle ซึ่งเป็นข้อมูลเกี่ยวกับธุรกิจอีคอมเมิร์ซ โดยมีเป้าหมายเพื่อวิเคราะห์พฤติกรรมลูกค้า คาดการณ์แนวโน้มยอดขาย และค้นหากลุ่มลูกค้าที่มีมูลค่าสูง

## 📂 Dataset
- **แหล่งข้อมูล**: [Online Retail Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset)
- **ข้อมูลที่มี**:
  - `InvoiceNo`: เลขที่ใบแจ้งหนี้
  - `StockCode`: รหัสสินค้า
  - `Description`: รายละเอียดสินค้า
  - `Quantity`: จำนวนที่ซื้อ
  - `InvoiceDate`: วันที่ออกใบแจ้งหนี้
  - `UnitPrice`: ราคาต่อหน่วย
  - `CustomerID`: รหัสลูกค้า
  - `Country`: ประเทศของลูกค้า

## 🎯 Goals & Objectives
1. **Data Cleaning & Preprocessing**: จัดการค่าข้อมูลที่หายไป และเตรียมข้อมูลให้เหมาะสมกับโมเดล
2. **Exploratory Data Analysis (EDA)**: วิเคราะห์แนวโน้มของลูกค้าและสินค้าขายดี
3. **Customer Segmentation**: แบ่งกลุ่มลูกค้าด้วยเทคนิค **K-Means Clustering** และ **DBSCAN**
4. **Sales Prediction**: (ลบออกเพื่อไม่ให้ขัดกับ Unsupervised Learning)

## 🔧 Technologies & Tools
- **Python** 🐍
- **Pandas, NumPy** (Data Processing)
- **Matplotlib, Seaborn, Plotly** (Data Visualization)
- **Scikit-Learn** (Machine Learning)
- **Kaggle API** (For downloading datasets)

## 🚀 Implementation Steps
1. **Data Preprocessing**
   - ตรวจสอบและลบค่าที่ขาดหาย
   - แปลงวันที่ให้เป็นฟอร์แมตที่เหมาะสม
   - คำนวณยอดขาย (TotalPrice = Quantity × UnitPrice)

```python
import pandas as pd

# Load dataset
retail = pd.read_excel("path/to/online_retail_II.xlsx")

# Drop missing values
retail = retail.dropna()

# Create TotalPrice column
retail["TotalPrice"] = retail["Quantity"] * retail["UnitPrice"]

# Convert InvoiceDate to datetime
retail["InvoiceDate"] = pd.to_datetime(retail["InvoiceDate"])
```

2. **Customer Segmentation (Clustering)**
   - ใช้ **K-Means** และ **DBSCAN** เพื่อแบ่งกลุ่มลูกค้า
   - คำนวณค่า Silhouette Score เพื่อเลือกจำนวนคลัสเตอร์ที่เหมาะสม

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Feature selection
X = retail.groupby("CustomerID").agg({"TotalPrice": "sum"}).reset_index()

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
X["Cluster"] = kmeans.fit_predict(X[["TotalPrice"]])

# Scatter plot of clusters
plt.scatter(X["CustomerID"], X["TotalPrice"], c=X["Cluster"], cmap="viridis")
plt.xlabel("Customer ID")
plt.ylabel("Total Spending")
plt.title("Customer Segmentation using K-Means")
plt.show()
```

## 📊 Results & Insights
- พบว่าบางกลุ่มสินค้ามียอดขายสูงมากในช่วงเทศกาล
- ลูกค้ามีพฤติกรรมการซื้อที่แตกต่างกัน โดยสามารถแบ่งออกเป็น **3-4 กลุ่มหลัก**
- โมเดล **K-Means** และ **DBSCAN** ช่วยในการแบ่งกลุ่มลูกค้าได้ดี

## 📌 Future Improvements
- ใช้ **Deep Learning** สำหรับการพยากรณ์ยอดขาย
- สร้าง **Recommendation System** เพื่อแนะนำสินค้าให้ลูกค้า
- วิเคราะห์พฤติกรรมลูกค้าแบบ Real-time ด้วย **Power BI**

## 📜 References
- [Online Retail Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset)
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [Pandas Official Documentation](https://pandas.pydata.org/)
- [Kaggle API Documentation](https://www.kaggle.com/docs/api)

## 🛠 How to Use
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset using Kaggle API:
   ```bash
   kaggle datasets download -d lakshmi25npathi/online-retail-dataset
   ```
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
