# 🛂 Online Retail Dataset - Machine Learning Project

## 📌 Overview
โปรเจกต์นี้เป็นการวิเคราะห์และพัฒนาโมเดล Machine Learning โดยใช้ **Online Retail Dataset** จาก Kaggle ซึ่งเป็นข้อมูลเกี่ยวกับธุรกิจอีคอมเมิร์ซ โดยมีเป้าหมายเพื่อวิเคราะห์พฤติกรรมลูกค้าและค้นหากลุ่มลูกค้าที่มีมูลค่าสูง

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
3. **Customer Segmentation**: แบ่งกลุ่มลูกค้าด้วยเทคนิค **RFM Analysis + K-Means Clustering**

## 🔧 Technologies & Tools
- **Python** 🐍
- **Pandas, NumPy** (Data Processing)
- **Matplotlib, Seaborn** (Data Visualization)
- **Scikit-Learn** (Machine Learning)
- **Power BI** (Dashboard & Visualization)

## 🚀 Implementation Steps
### 1. Data Preprocessing
- ตรวจสอบและลบค่าที่ขาดหาย
- แปลงวันที่ให้เป็นฟอร์แมตที่เหมาะสม
- คำนวณยอดขาย (`TotalPrice = Quantity × UnitPrice`)

### 2. Exploratory Data Analysis (EDA)
- วิเคราะห์ยอดขายรายวัน/เดือน
- ค้นหาสินค้าขายดีและลูกค้ารายใหญ่
- วาดกราฟแสดง Distribution ของข้อมูล

### 3. Customer Segmentation (RFM Analysis)
RFM Analysis เป็นเทคนิคที่ใช้แบ่งกลุ่มลูกค้าตามพฤติกรรมการซื้อ โดยพิจารณาจาก:
- **Recency (R)**: ลูกค้าซื้อสินค้าล่าสุดเมื่อไหร่
- **Frequency (F)**: ลูกค้าซื้อบ่อยแค่ไหน
- **Monetary (M)**: ลูกค้าใช้จ่ายทั้งหมดเท่าไหร่
  
![RFM Clustering](https://raw.githubusercontent.com/Pariman1419/online_retail_II/main/newplot.png)

**การแบ่งกลุ่มด้วย K-Means Clustering**
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm_data["Cluster"] = kmeans.fit_predict(rfm_data)





