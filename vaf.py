import joblib
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

st.header("จำแนกผักและผลไม้")
st.header("ผักและผลไม้ในเว็บนี้"
          "เราจะใช้ความหวานและความกรอบในการแยกความแตกต่าง")

st.sidebar.header("ตัวอย่างคะแนน")
st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSdAK4iWyrvEx7PZJWquHUIzDFoEy10J2MjRQ&usqp=CAU")

tab1, tab2 = st.sidebar.columns(2)
with tab1:
    st.selectbox("sweetness vegetables", ["carrot : 2",
                                          "bean : 1",
                                          "cabbage : 2",
                                          "chili : 0",
                                          "cucumber : 2",
                                          "garlic : 0",
                                          "lettuce : 1",
                                          "long bean : 2",
                                          "onion : 4",
                                          "tomato : 2"])
    st.selectbox("crunchiness vegetables", ["carrot : 8",
                                            "bean : 4",
                                            "cabbage : 8",
                                            "chili : 0",
                                            "cucumber : 7",
                                            "garlic : 0",
                                            "lettuce : 8",
                                            "long bean : 3",
                                            "onion : 4",
                                            "tomato : 0"])
with tab2:
    st.selectbox("sweetness fruits", ["apple : 9",
                                      "banana : 10",
                                      "coconut : 9",
                                      "grape : 6",
                                      "strawberry : 7",
                                      "orange : 6"])
    st.selectbox("crunchiness fruits", ["apple : 4",
                                        "banana : 0",
                                        "coconut : 0",
                                        "grape : 3",
                                        "strawberry : 0",
                                        "orange : 0"])

st.sidebar.image("https://st.depositphotos.com/1793489/1946/v/950/depositphotos_19468125-stock-illustration-cartoon-fruits-and-vegetables-with.jpg")

# https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/A_black_image.jpg/640px-A_black_image.jpg
# https://picsum.photos/960/720?blur=2
#https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/assortment-of-colorful-ripe-tropical-fruits-top-royalty-free-image-995518546-1564092355.jpg?crop=0.982xw:0.736xh;0,0.189xh&resize=980:*
st.markdown(
    f"""
       <style>
       .stApp {{
           background-image: url("https://png.pngtree.com/background/20210711/original/pngtree-summer-atmosphere-restaurant-supermarket-vegetable-psd-layered-promotion-background-picture-image_1098004.jpg");
           background-attachment: fixed;
           background-size: cover;
           /* opacity: 0.3; */
       }}
       </style>
       """,
    unsafe_allow_html=True
)

left, right = st.columns(2)
left.markdown("เว็บนี้ให้คำตอบว่าสิ่งๆนั้นเป็นผักหรือผลไม้?")
left.markdown("โดยที่ผู้ใช้กรอกความหวานและความกรอบเป็น : ตัวเลข 1 ถึง 10")

df = pd.read_csv('vegetables_and_fruits01.csv')
xx = df[['Sweetness', 'Crunchiness']]
yy = df['Type']

X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

pred = (2, 5)
pred1 = np.asarray(pred)
pred2 = pred1.reshape(1, -1)
realpre = clf.predict(pred2)

cols = st.columns(2)
with cols[0]:
    numbers = st.slider("ENTER sweetness stat", 0, 10)
with cols[1]:
    number = st.slider("ENTER crunchiness stat", 0, 10)

a = st.button("ENTER")
if a:
    c = clf.predict([[numbers, number]])
    st.header(c)

def generate_vegetables_and_fruits_data():
    df = pd.read_csv('vegetables_and_fruits01.csv')
    x = df[['Sweetness', 'Crunchiness']]
    y = df['Type']
    df_new = pd.DataFrame({
        'Sweetness': x['Sweetness'],
        'Type': y,
        'Crunchiness': x['Crunchiness']
    })
    df_new.to_csv('vegetables_and_fruits.csv', index=False)


def load_vegetables_and_fruits_data():
    return pd.read_csv('vegetables_and_fruits01.csv')

def save_model(model):
    joblib.dump(model, 'model.joblib')

def load_model():
    return joblib.load('model.joblib')

generate = right.button('generate vegetables_and_fruits.csv')
if generate:
    right.write('generating "vegetables_and_fruits.csv" ...')
    generate_vegetables_and_fruits_data()
    right.write(' ... done')

load = right.button('vegetables_and_fruits.csv')
if load:
    right.write('loading "vegetables_and_fruits.csv ..."')
    df = pd.read_csv('vegetables_and_fruits01.csv', index_col=0)
    right.write('... done')
    right.dataframe(df)


train = right.button('train vegetables_and_fruits')
if train:
    right.write('training model ')
    df = pd.read_csv('vegetables_and_fruits.csv', index_col=0)
    model = LinearRegression()
    right.write('')
    right.dataframe(df)
    save_model(model)

















