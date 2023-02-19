import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# โหลดข้อมูล iris dataset จาก scikit-learn
iris = datasets.load_iris()

# สร้างตัวแปร X และ y เพื่อเก็บ features และ target ของ iris dataset
X = iris.data
y = iris.target

# สร้างโมเดล Random Forest classifier
clf = RandomForestClassifier()

# ฝึกโมเดลด้วยข้อมูล iris dataset
clf.fit(X, y)

# สร้างหน้าเว็บด้วย Streamlit
st.write("""
# แบบทดสอบเดาเพศของผู้ใช้งาน
""")
st.write("โปรดกรอกข้อมูลด้านล่างเพื่อทำการเดาเพศของผู้ใช้งาน")

# สร้างฟอร์มเพื่อรับข้อมูลจากผู้ใช้
sepal_length = st.text_input(f'ส่วนสูง {int(input())}')
sepal_width = st.slider('ความกว้างกลีบเลี้ยง (cm)', 2.0, 4.5, 3.4)
petal_length = st.slider('ความยาวกลีบดอก (cm)', 1.0, 7.0, 1.3)
petal_width = st.slider('ความกว้างกลีบดอก (cm)', 0.1, 2.5, 0.2)

# นำข้อมูลที่ผู้ใช้กรอกเข้ามาใช้งานโมเดล
gender_prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
gender = iris.target_names[gender_prediction[0]]

st.write("เพศของผู้ใช้งานคือ", gender)