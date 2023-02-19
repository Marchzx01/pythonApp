import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
st.markdown(
    f"""
       <style>
       .stApp {{
           background-image: url("https://picsum.photos/960/720?blur=2");
           background-attachment: fixed;
           background-size: cover;
           /* opacity: 0.3; */
       }}
       </style>
       """,
    unsafe_allow_html=True
)
st.title('House Price Prediction')
#st.sidebar.title('Menu')
left, right = st.columns(2)
left.markdown("เว็บนี้ให้คำตอบว่าสิ่งๆนั้นเป็นผักหรือผลไม้?")
left.markdown("โดยที่ผู้ใช้กรอกความหวานและความกรอบเป็น : ตัวเลข")
#right.markdown('![image](https://picsum.photos/id/555/200/300/)')
def generate_house_data():
    rng = np.random.RandomState(0)
    n = 10
    x = np.round(400*rng.rand(n), -1).astype(np.int16) # 'พื้นที่(ตรว)'
    y = np.round(40000*x + 20000 + 200000*rng.rand(n), -4).astype(np.int32) # ราคา(บาท)
    df = pd.DataFrame({
        'x': x,
        'y': y
    })
    df.to_excel('house.xlsx')

def load_house_data():
    return pd.read_excel('house.xlsx')

def save_model(model):
    joblib.dump(model, 'model.joblib')

def load_model():
    return joblib.load('model.joblib')

generateb = right.button('generate house.xlsx')
if generateb:
    right.write('generating "house.xlsx" ...')
    generate_house_data()
    right.write(' ... done')

loadb = right.button('load house.xlsx')
if loadb:
    right.write('loading "house.xlsx ..."')
    df = pd.read_excel('house.xlsx', index_col=0)
    right.write('... done')
    right.dataframe(df)
    fig, ax = plt.subplots()
    df.plot.scatter(x='x', y='y', ax=ax)
    st.pyplot(fig)

trainb = right.button('train แบบจำลองประเมินราคา')
if trainb:
    right.write('training model ...')
    df = pd.read_excel('house.xlsx', index_col=0)
    model = LinearRegression()
    model.fit(df.x.values.reshape(-1,1), df.y)
    right.write('... done')
    right.dataframe(df)
    save_model(model)

area = left.number_input('พื้นที่(ตรว.)')
predictb = left.button('ประเมินราคา')
if predictb:
    #predict = 2000000 + 40000*area
    model = load_model()
    predict = model.predict(np.array([area]).reshape(-1,1))
    left.markdown(f'พื้นที่บ้าน :green[{area} ตรว.] ควรจะขาย :red[{predict[0]:,.2f} บาท]')





