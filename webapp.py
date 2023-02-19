import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st

# อ่านข้อมูล Dataset จากไฟล์ CSV
df = pd.read_csv('fruits_veggies.csv')

# แสดงข้อมูล Dataset
st.title('Fruits and Vegetables Dataset')
st.write(df)

# เลือกคุณสมบัติของผักและผลไม้ที่จะนำมาใช้ในการแยกแบ่ง
features = ['sweetness', 'crunchiness']

# สร้างตัวแบ่ง K-means แบ่ง Dataset เป็น 2 กลุ่ม
kmeans = KMeans(n_clusters=2)

# สร้าง Dataset ที่เลือกคุณสมบัติมา
X = df[features]

# ฝึกโมเดลแบ่งกลุ่มด้วย K-means
kmeans.fit(X)

# แสดง Label ของแต่ละชนิดผักและผลไม้
labels = kmeans.labels_

#
x = st.text_input('NAME')
# สร้าง function สำหรับแสดงผลลัพธ์
def show_results(labels):
    for i, label in enumerate(labels):
        name = df.loc[i+1, 'name']
        if label == 0:
            st.write(f'{name} is a vegetable')
        else:
            st.write(f'{name} is a fruit')



# แสดงผลลัพธ์
st.title('Fruits and Vegetables Classifier')
st.write('Enter the sweetness and crunchiness values of the fruit or vegetable you want to classify:')

sweetness = st.slider('Sweetness', 0, 10, 5)
crunchiness = st.slider('Crunchiness', 0, 10, 5)



# ใช้โมเดล K-means Clustering ที่ฝึกกับ Dataset เพื่อทำนายว่าเป็นผักหรือผลไม้
input_features = [[sweetness, crunchiness]]
label = kmeans.predict(input_features)

# แสดงผลลัพธ์
show_results(label)
