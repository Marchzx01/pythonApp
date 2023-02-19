import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# อ่านไฟล์ CSV
df = pd.read_csv("vegetables_and_fruits - ชีต1.csv")
# Features คือคอลัมน์ 'Sweetness' และ 'Crunchiness'
X = df[['Sweetness', 'Crunchiness']]

# Target คือคอลัมน์ 'Type'
y = df['Type']


# แบ่งข้อมูลออกเป็น Train set และ Test set โดยสุ่มเลือก 20% เป็น Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# สร้างโมเดล Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train โมเดลด้วย Train set
clf.fit(X_train, y_train)
# ทดสอบโมเดลด้วย Test set
y_pred = clf.predict(X_test)

# คำนวณความแม่นยำ


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# ใช้โมเดลทำนายผลของข้อมูลใหม่
new_data = pd.DataFrame({'Sweetness': [8, 4], 'Crunchiness': [5, 9]})
predictions = clf.predict(new_data)

# แสดงผลลัพธ์การทำนาย
for i in range(len(predictions)):
    print("Predicted type for new data point", i+1, ":", predictions[i])

# สร้างหน้าเว็บแอปพลิเคชัน

    # กำหนดชื่อหน้าเว็บแอปพลิเคชัน
    st.title("Vegetable and Fruit Sweetness and Crunchiness Predictor")

    # สร้างเนื้อหาในหน้าเว็บแอปพลิเคชัน
    st.write("This app can predict the sweetness and crunchiness of vegetables and fruits based on their properties.")

    # รับค่าข้อมูลเข้า
    vegetable_or_fruit = st.selectbox("Select vegetable or fruit", ["Vegetable", "Fruit"])
    if vegetable_or_fruit == "Vegetable":
        color = st.slider("Color (1 = light green, 10 = dark green)", 1, 10)
        shape = st.slider("Shape (1 = round, 10 = long)", 1, 10)
        size = st.slider("Size (1 = small, 10 = large)", 1, 10)
        texture = st.slider("Texture (1 = soft, 10 = hard)", 1, 10)
        data = [[color, shape, size, texture]]
    else:
        color = st.slider("Color (1 = light yellow, 10 = dark yellow)", 1, 10)
        shape = st.slider("Shape (1 = round, 10 = long)", 1, 10)
        size = st.slider("Size (1 = small, 10 = large)", 1, 10)
        texture = st.slider("Texture (1 = soft, 10 = hard)", 1, 10)
        data = [[color, shape, size, texture]]

    # ทำนายค่า
    prediction = clf.predict(data)

    # แสดงผลลัพธ์
    if prediction == 1:
        st.write("This vegetable or fruit is sweet and crunchy.")
    elif prediction == 2:
        st.write("This vegetable or fruit is sweet but not crunchy.")
    elif prediction == 3:
        st.write("This vegetable or fruit is not sweet but is crunchy.")
    else:
        st.write("This vegetable or fruit is not sweet and not crunchy.")





