import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Hello Streamlit
# -----------------------------
st.title("Hello Streamlit 👋")
st.write("This is your first Streamlit app!")

name = st.text_input("Enter your name:")

if name:
    st.success(f"Hello {name}, welcome to Streamlit!")

# -----------------------------
# Text & Titles
# -----------------------------
st.title("Main Title")
st.header("Header")
st.subheader("Subheader")
st.text("Simple text")
st.markdown("**Bold text** with *italics* and `code`")

# -----------------------------
# Widgets
# -----------------------------
if st.button("Click Me"):
    st.write("Button clicked!")

agree = st.checkbox("I agree")

if agree:
    st.write("✅ You agreed!")

choice = st.radio("Choose one:", ["Option 1", "Option 2"])
st.write("You selected:", choice)

option = st.selectbox("Pick a number:", [1, 2, 3, 4, 5])
st.write("Your number is:", option)

value = st.slider("Select a range", 0, 100, 25)
st.write("Slider value:", value)

# -----------------------------
# Data Display
# -----------------------------
df = pd.DataFrame(
    np.random.randn(10, 3),
    columns=["A", "B", "C"]
)

st.write("Random DataFrame:")
st.write(df)

st.line_chart(df)
st.bar_chart(df)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

# -----------------------------
# Sidebar & Layout
# -----------------------------
st.sidebar.title("Sidebar Menu")

user = st.sidebar.text_input("Enter your username")
st.sidebar.write("Welcome", user)

col1, col2 = st.columns(2)

with col1:
    st.write("This is column 1")

with col2:
    st.write("This is column 2")

# -----------------------------
# Iris Flower Prediction
# -----------------------------
st.title("🌸 Iris Flower Prediction")

iris = load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])

st.write("Prediction:", iris.target_names[prediction][0])