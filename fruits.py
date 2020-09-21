import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
import streamlit as st
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time
st.title('Welcome to the fruit type prediction application')
fruits = pd.read_excel(r'C:\adrian kasito\adrian\fruit_data_with_colors.xlsx')




st.write("This is an application for predicting the fruit type using machine learning. Let's try and see!")
image = Image.open(r'C:\Users\Mr. Capable\Documents\fruits_data_science\pic.jpg')
st.image(image, width='50%', caption='Fruits', use_column_width=True)
check_data = st.checkbox("See the sample data")
if check_data:
  st.write(fruits)
st.write("Now let's find out the fruit_mass, fruit_width, fruit_height and fruit_color_score")

fruit_mass    = st.slider("What is the fruit mass in grams?",int(fruits.mass.min()),int(fruits.mass.max()),int(fruits.mass.mean()) )
fruit_width    = st.slider("What is the fruit_width in centimeters?",float(fruits.width.min()),float(fruits.width.max()),float(fruits.width.mean()) )
fruit_height = st.slider('What is the fruit height in centimeters',float(fruits.height.min()),float(fruits.height.max()),float(fruits.height.mean()) )
fruit_color_score = st.slider('What is the fruit_color_score',float(fruits.color_score.min()), float(fruits.color_score.max()))

#splitting your data
# Create train_test_split
X = fruits[['mass', 'width', 'height', 'color_score']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=2)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 5)


#fitting and predict your model
model.fit(X_train, y_train)
model.predict(X_test)
errors = np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
predictions = model.predict([[fruit_mass, fruit_width, fruit_height, fruit_color_score]])[0]

#checking prediction house price
if st.button("Run me!"):
  st.header("Your fruit_name is {} ".format(int(predictions)))
fruit_type = st.radio('Which fruit have you predicted?', (1,2,3,4))
if fruit_type == 1:
    st.write('You predicted an apple')
elif fruit_type == 2:
    st.write('You predicted a  madarin')
elif fruit_type == 3:
    st.write('You have predicted an orange')
else:
    st.write('You predicted a lemon')

st.balloons()
