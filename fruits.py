import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

plt.rcParams['figure.figsize'] = (20.0, 10.0)
import streamlit as st
from PIL import Image

st.title('Welcome to the fruit type prediction application')
fruits = pd.read_excel('fruit_data_with_colors.xlsx')

st.write("This is an application for predicting the fruit type using machine learning. Let's try and see!")
image = Image.open('pic.jpg')
st.image(image, width='50%', caption='Fruits', use_column_width=True)
check_data = st.checkbox("See the sample data")
if check_data:
    st.write(fruits)

st.write("Now let's find out the fruit_mass, fruit_width, fruit_height and fruit_color_score")

fruit_mass = st.slider("What is the fruit mass in grams?", int(fruits.mass.min()), int(fruits.mass.max()),
                       int(fruits.mass.mean()))
fruit_width = st.slider("What is the fruit_width in centimeters?", float(fruits.width.min()), float(fruits.width.max()),
                        float(fruits.width.mean()))
fruit_height = st.slider('What is the fruit height in centimeters', float(fruits.height.min()),
                         float(fruits.height.max()), float(fruits.height.mean()))
fruit_color_score = st.slider('What is the fruit_color_score', float(fruits.color_score.min()),
                              float(fruits.color_score.max()))

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(fruits[['mass', 'width', 'height', 'color_score']])

# Splitting your data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, fruits['fruit_label'], test_size=0.15, random_state=2)

# Create and fit KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Calculate RMSE
errors = np.sqrt(mean_squared_error(y_test, predictions))

# Checking prediction fruit type
if st.button("Run me!"):
    fruit_data = scaler.transform([[fruit_mass, fruit_width, fruit_height, fruit_color_score]])
    predictions = model.predict(fruit_data)[0]
    st.header("Your fruit_name is {} ".format(int(predictions)))

fruit_type = st.radio('Which fruit have you predicted?', (1, 2, 3, 4))
if fruit_type == 1:
    st.write('You predicted an apple')
elif fruit_type == 2:
    st.write('You predicted a mandarin')
elif fruit_type == 3:
    st.write('You have predicted an orange')
else:
    st.write('You predicted a lemon')

st.balloons()
