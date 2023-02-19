import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import streamlit as st

# Set the title and the image for the app
st.title("Gender Classification App")
st.image("logo.png")

# Add a button to take a photo with the webcam
if st.button("Take a photo"):
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    img = cv2.resize(img, (50, 50))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_data = img_gray.flatten()
    gender = clf.predict([img_data])[0]
    cap.release()
    st.write(f"Predicted gender: {gender}")
    st.image(img, caption="Captured photo")


# Create a list to store the image data and gender labels
images = []
labels = []

# Read in the image data and gender labels
for i in range(1, 1001):
    img_path = "images/" + str(i) + ".jpg"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (50, 50))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img_gray.flatten())

    if i <= 500:
        labels.append(0)
    else:
        labels.append(1)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create a SVM model and fit the data
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



