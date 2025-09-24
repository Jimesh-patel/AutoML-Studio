import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

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

# Session State Initialization
if "df_original" not in st.session_state:
    st.session_state.df_original = None
if "df_copy" not in st.session_state:
    st.session_state.df_copy = None

if uploaded_file:
    if st.session_state.df_original is None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df_original = df.copy()
        st.session_state.df_copy = df.copy()

    df = st.session_state.df_original
    df_copy = st.session_state.df_copy

    st.success("✅ File uploaded successfully!")

    if st.button("Reset"):
        st.session_state.clear()
        try:
            st.rerun()
        except:
            st.markdown("""
                <meta http-equiv="refresh" content="0">
            """, unsafe_allow_html=True)
            
    # Preview
    
    st.markdown("### Preview of Dataset")
    st.write(f"Shape : {df.shape}")
    st.dataframe(df.head())
    st.markdown("### Summary Statistics")
    st.dataframe(df.describe())
    st.markdown("---")
        
        
    # --- EDA Section ---
    st.header("Explore & Understand Your Data (EDA)")
  

    st.subheader("Select a Column to Explore")
    selected_col = st.selectbox("Choose a column", df.columns)

    st.markdown("**Visual Options**")
    show_corr = st.checkbox("Show Correlation Matrix (Numeric Only)")
    show_missing = st.checkbox("Show Missing Value Heatmap")

    st.subheader("Data Insights")

    if selected_col:
        if df[selected_col].dtype in ["float64", "int64"]:
            st.markdown(f"**Numeric Distribution: `{selected_col}`**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Histogram")
                fig1, ax1 = plt.subplots(figsize= (7, 4))
                sns.histplot(df[selected_col].dropna(), kde=True, ax=ax1)
                ax1.set_xlabel(selected_col)
                ax1.set_title(f"Histogram of {selected_col}")
                st.pyplot(fig1)

            with col2:
                st.markdown("### Boxplot")
                fig2, ax2 = plt.subplots(figsize = (7, 3.65))
                sns.boxplot(x=df[selected_col], ax=ax2)
                ax2.set_xlabel(selected_col)
                ax2.set_title(f"Boxplot of {selected_col}")
                st.pyplot(fig2)
        else:
            st.markdown(f"**Categorical Value Counts: `{selected_col}`**")
            st.markdown("### Bar Chart")
            value_counts = df[selected_col].value_counts().reset_index()
            value_counts.columns = [selected_col, "count"]
            fig = px.bar(value_counts, x=selected_col, y="count", labels={selected_col: selected_col, "count": "Count"})
            fig.update_layout(title=f"Bar Chart of {selected_col}")
            st.plotly_chart(fig)
            
    col1, col2 = st.columns(2)
    with col1:
        if show_corr:
            num_df = df.select_dtypes(include=[np.number])
            if not num_df.empty:
                st.markdown("### Correlation Matrix")
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                sns.heatmap(
                    num_df.corr(),
                    annot=True,
                    cmap="coolwarm",
                    ax=ax3,
                    fmt=".2f",  
                    annot_kws={"size": 8}  
                )
                ax3.set_title("Correlation Matrix")
                st.pyplot(fig3)
            else:
                st.warning("No numeric columns found to compute correlation matrix.")
    with col2:
        if show_missing:
            st.markdown("### Missing Values Heatmap")
            fig4, ax4 = plt.subplots(figsize=(8, 4.7))
            sns.heatmap(df.isnull(), cbar=False, cmap="Blues", ax=ax4)
            ax4.set_title("Missing Value Heatmap")
            st.pyplot(fig4)

    st.info("You've completed the EDA section. Now you can move to preprocessing!")
    st.markdown("---")



