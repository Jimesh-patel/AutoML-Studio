import streamlit as st
import pandas as pd

# Page Config
st.set_page_config(page_title="AutoML-Studio", layout="wide")

# Main Title
st.markdown(
    "<h1 style='text-align: center; color: #4F8BF9;'>AutoML-Studio</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center; color: #333;'>Machine Learning Dashboard</h3>",
    unsafe_allow_html=True
)

st.markdown("---")

# Upload Section
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.write("### Data Preview")
        st.dataframe(df.head(20))
        st.write("**Shape:**", df.shape)
        st.write("**Columns:**", df.columns.tolist())
        st.write("### Summary Statistics")
        st.dataframe(df.describe())
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
else:
    st.info("Please upload a CSV file to get started.")

