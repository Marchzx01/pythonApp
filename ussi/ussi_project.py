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
left.markdown("เว็บนี้ให้คำตอบว่าควรขายบ้านราคาเท่าใด?")
left.markdown("โดยที่ผู้ใช้กรอกขนาดพื้นที่ของบ้าน หน่วย ตารางวา")
#right.markdown('![image](https://picsum.photos/id/555/200/300/)')
def generate_house_data():
    f= pd.read_excel("./ussi/LinearRegression.xlsx",usecols="A",nrows=11,skiprows=1,index_col=None,header=0)
    x=[]
    x.append(f(index=False))
    # x = np.array(x).astype(np.int16)

    c= pd.read_excel("./ussi/LinearRegression.xlsx",usecols="D",nrows=11,skiprows=1,index_col=None,header=0)
    y=[]
    y.append(c(index=False))

    # y=np.array(y).astype(np.int32)
    rng = np.random.RandomState(0)
    n = 10
    # x = np.round(400*rng.rand(n), -1).astype(np.int16) # 'พื้นที่(ตรว)'
    # y = np.round(40000*x + 20000 + 200000*rng.rand(n), -4).astype(np.int32) # ราคา(บาท)
    st.markdown(x)
    st.markdown(y)
    df = pd.DataFrame({
        'x': x,
        'y': y
    })
    df.to_excel('./ussi/data.xlsx')

def load_house_data():
    return pd.read_excel('./data.xlsx')

def save_model(model):
    joblib.dump(model, './train.joblib')

def load_model():
    return joblib.load('./train.joblib')

generateb = right.button('generate house.xlsx')
if generateb:
    right.write('generating "data.xlsx" ...')
    generate_house_data()
    right.write(' ... done')

loadb = right.button('load data.xlsx')
if loadb:
    right.write('loading "data.xlsx ..."')
    df = pd.read_excel("./ussi/data.xlsx",header=0, names=['co', 'co'] ,index_col=None)
    right.write('... done')
    right.dataframe(df)
    fig, ax = plt.subplots()
    df.plot.scatter(x='co', y='co.1', ax=ax)
    st.pyplot(fig)

trainb = right.button('train แบบจำลองประเมินราคา')
if trainb:
    right.write('training model ...')
    df = pd.read_excel("./ussi/data.xlsx",header=0, names=['co', 'co'] ,index_col=None)
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