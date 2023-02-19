import joblib
import numpy as np
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression


# Styling
primaryColor = "green"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"

st.header("Vegetables and Fruits Classification")
st.subheader("We use Sweetness and Crunchiness to classify vegetables and fruits")

# Selectboxes
tab1, tab2 = st.sidebar.columns(2)
with tab1:
    sweetness_veg = st.selectbox("Sweetness of Vegetables", range(11))
    crunchiness_veg = st.selectbox("Crunchiness of Vegetables", range(11))
with tab2:
    fruit_options = {
        "Apple": 9,
        "Banana": 10,
        "Coconut": 9,
        "Grape": 6,
        "Strawberry": 7,
        "Orange": 6,
    }
    fruit = st.selectbox("Fruit", options=list(fruit_options.keys()))
    sweetness_fruit = fruit_options[fruit]
    crunchiness_fruit = st.selectbox("Crunchiness of Fruit", range(11))

# Background image
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("https://static.hd.co.th/989x504/system/blog_articles/main_hero_images/000/001/456/original/iStock-506149410_L.jpg");
            background-attachment: fixed;
            background-size: cover;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

left, right = st.columns(2)
left.markdown("This website helps you to classify whether something is a fruit or a vegetable.")
left.markdown("You need to provide the sweetness and crunchiness values as numbers.")

# Load data
@st.cache
def load_data():
    return pd.read_csv("vegetables_and_fruits - ชีต1.csv")

df = load_data()

# Train model
@st.cache
def train_model():
    X = df[["Sweetness", "Crunchiness"]]
    y = df["Type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf

clf = train_model()

# Predict the type of given values
def predict_type(sweetness, crunchiness):
    pred = clf.predict(np.array([[sweetness, crunchiness]]))
    return pred[0]

# Predict type of input values
if st.button("ENTER"):
    if fruit:
        sweet = sweetness_fruit
        crunch = crunchiness_fruit
    else:
        sweet = sweetness_veg

right.write("regressor created using LinearRegression()")
regressor = LinearRegression()

X = df['Sweetness'].values.reshape(-1,1)
y = df['Crunchiness'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor.fit(X_train, y_train)

right.write("regressor trained!")

save_modelb = right.button('save model')
if save_modelb:
    right.write('saving regressor to "regressor.joblib" ...')
    save_modelb(regressor)
    right.write(' ... done')

load_modelb = right.button('load model')
if load_modelb:
    right.write('loading regressor from "regressor.joblib" ...')
    regressor = load_modelb()
    right.write(' ... done')

def make_prediction(sweetness):
 return regressor.predict([[sweetness]])

c, d = st.columns(2)
c.header("ความหวาน")
sweetness = c.slider("", min_value=0, max_value=10)
d.header("ความกรอบ")
crunchiness = d.number_input("", min_value=0, max_value=10)

if sweetness:
    prediction = make_prediction(sweetness)
    st.header("ความกรอบที่คาดการณ์ได้คือ")
    st.write(prediction[0][0])
# Plot data and regression line
    fig, ax = plt.subplots()
    ax.scatter(X, y)
    ax.plot(X, regressor.predict(X), color='red')
    ax.set_xlabel('Sweetness')
    ax.set_ylabel('Crunchiness')
    ax.set_title('Sweetness vs. Crunchiness')
    st.pyplot(fig)