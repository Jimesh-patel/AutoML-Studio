import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import io
import pickle
from sklearn.model_selection import GridSearchCV



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

    st.success("File uploaded successfully!")

    if st.button("Reset"):
        st.session_state.clear()
        try:
            st.rerun()
        except:
            st.markdown("""
                <meta http-equiv="refresh" content="0">
            """, unsafe_allow_html=True)
            
# --- Preview ---
    
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
    
# ---Feature Extraction ---
    st.markdown("## Feature Selection")
    with st.expander("Drop Unwanted Columns"):
        drop_cols = st.multiselect("Select columns to drop from dataset", st.session_state.df_copy.columns.tolist(), key="drop_cols")
        if st.button("Drop Selected Columns"):
            st.session_state.df_copy.drop(columns=drop_cols, inplace=True)
            st.success(f"Dropped columns: {', '.join(drop_cols)}")
            st.dataframe(st.session_state.df_copy.head())

    num_cols = st.session_state.df_copy.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        if st.button("Show Correlation Heatmap"):
            corr_matrix = st.session_state.df_copy[num_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 3))  
            sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f",cbar=True)
            plt.xticks(fontsize=8, rotation=0)  
            plt.yticks(fontsize=8)   
            st.pyplot(fig)
    else:
        st.info("No numerical features to generate correlation heatmap.")
    
    st.markdown("---")
    
# --- Identify Missing Columns ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_missing_cols = [col for col in numeric_cols if col in df_copy.columns and df_copy[col].isnull().sum() > 0]
    categorical_missing_cols = [col for col in categorical_cols if col in df_copy.columns and df_copy[col].isnull().sum() > 0]

# --- Handle Missing Numerical Data ---
    st.markdown("## Handle Missing Data")
    with st.expander("Handle Missing Numerical Data"):
        if numeric_missing_cols:
            selected_num_cols = st.multiselect("Select Numeric Columns to Impute", numeric_missing_cols)

            imputer_type = st.selectbox(
                "Choose Imputation Strategy",
                ("Mean", "Median", "Most Frequent", "KNN Imputer")
            )

            if imputer_type != "KNN Imputer":
                strategy_map = {
                    "Mean": "mean",
                    "Median": "median",
                    "Most Frequent": "most_frequent"
                }
                strategy = strategy_map[imputer_type]
                if st.button("Impute Numeric Data"):
                    try:
                        imp = SimpleImputer(strategy=strategy)
                        st.session_state.df_copy[selected_num_cols] = imp.fit_transform(st.session_state.df_copy[selected_num_cols])
                        st.success(f"Missing values imputed using '{strategy}' strategy.")
                        st.dataframe(st.session_state.df_copy[selected_num_cols].head())
                    except Exception as e:
                        st.warning(f"Error: {str(e)}")
            else:
                neighbors = st.slider("Select number of neighbors for KNN", min_value=1, max_value=10, value=3)
                if st.button("Impute using KNN"):
                    try:
                        imp = KNNImputer(n_neighbors=neighbors)
                        st.session_state.df_copy[selected_num_cols] = imp.fit_transform(st.session_state.df_copy[selected_num_cols])
                        st.success(f"Missing values imputed using KNN with {neighbors} neighbors.")
                        st.dataframe(st.session_state.df_copy[selected_num_cols].head())
                    except Exception as e:
                        st.warning(f"Error: {str(e)}")
        else:
            st.info("✅ No missing values found in numeric columns.")

    # --- Handle Missing Categorical Data ---
    with st.expander("Handle Missing Categorical Data"):
        if categorical_missing_cols:
            selected_cat_cols = st.multiselect("Select Categorical Columns to Impute", categorical_missing_cols)

            cat_strategy = st.selectbox(
                "Choose Imputation Strategy for Categorical Columns",
                ("Most Frequent", "Constant Value")
            )

            if cat_strategy == "Most Frequent":
                if st.button("Impute Categorical (Most Frequent)"):
                    try:
                        imp = SimpleImputer(strategy='most_frequent')
                        st.session_state.df_copy[selected_cat_cols] = imp.fit_transform(
                            st.session_state.df_copy[selected_cat_cols])
                        st.success("Missing categorical values filled with most frequent value.")
                        st.dataframe(st.session_state.df_copy[selected_cat_cols].head())
                    except Exception as e:
                        st.warning(f"Error: {str(e)}")
            else:
                constant_val = st.text_input("Enter constant value to replace missing data", value="Unknown")
                if st.button("Impute Categorical (Constant Value)"):
                    try:
                        imp = SimpleImputer(strategy='constant', fill_value=constant_val)
                        st.session_state.df_copy[selected_cat_cols] = imp.fit_transform(
                            st.session_state.df_copy[selected_cat_cols])
                        st.success(f"Missing values replaced with constant: '{constant_val}'.")
                        st.dataframe(st.session_state.df_copy[selected_cat_cols].head())
                    except Exception as e:
                        st.warning(f"Error: {str(e)}")
        else:
            st.info("No missing values found in categorical columns.")
            
    # --- Final Preview of Cleaned Dataset ---
    st.markdown("---")
    st.header("Updated Dataset Preview (After Imputation)")
    st.dataframe(st.session_state.df_copy.head(20))
    st.success("This is your cleaned dataset after all missing values have been handled.")

    # --- Reset Button ---
    if st.button("Reset Changes"):
        st.session_state.df_copy = st.session_state.df_original.copy()
        st.success("Dataset has been reset to original state.")
        
    st.markdown("---")
    
# --- Scaling & Encoding ---
    
    st.header("Scaling & Encoding")

    # Create copy for X (without target for now)
    X = st.session_state.df_copy.copy()

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    with st.expander("Scale Numerical Features"):
        if num_cols:
            selected_scale_cols = st.multiselect("Select Numeric Columns to Scale", num_cols, key="scale_cols")

            scaler_option = st.selectbox("Choose a Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler"])

            if st.button("Scale Selected Columns"):
                if selected_scale_cols:
                    scaler_map = {
                        "StandardScaler": StandardScaler(),
                        "MinMaxScaler": MinMaxScaler(),
                        "RobustScaler": RobustScaler()
                    }

                    try:
                        scaler = scaler_map[scaler_option]
                        scaled_values = scaler.fit_transform(X[selected_scale_cols])
                        st.session_state.df_copy[selected_scale_cols] = scaler.fit_transform(X[selected_scale_cols])
                        st.success(f"Columns scaled using {scaler_option}.")
                        st.dataframe(st.session_state.df_copy[selected_scale_cols].head())
                    except Exception as e:
                        st.warning(f"Scaling Error: {str(e)}")
                else:
                    st.warning("Please select at least one column to scale.")
        else:
            st.warning("No numeric columns to scale.")

    with st.expander("Encode Categorical Features"):
        if cat_cols:
            selected_encode_cols = st.multiselect("Select Categorical Columns to Encode", cat_cols, key="encode_cols")

            encoding_type = st.selectbox("Choose Encoding Method", ["Label Encoding", "One-Hot Encoding"])

            if st.button("Encode Selected Columns"):
                if selected_encode_cols:
                    try:
                        if encoding_type == "Label Encoding":
                            for col in selected_encode_cols:
                                le = LabelEncoder()
                                st.session_state.df_copy[col] = le.fit_transform(st.session_state.df_copy[col].astype(str))
                            st.success("Label Encoding applied.")
                            st.dataframe(st.session_state.df_copy[selected_encode_cols].head())

                        elif encoding_type == "One-Hot Encoding":
                            try:
                                original_columns = st.session_state.df_copy.columns.tolist()
                                st.session_state.df_copy = pd.get_dummies(
                                    st.session_state.df_copy, columns=selected_encode_cols, drop_first=True
                                )

                                new_columns = [col for col in st.session_state.df_copy.columns if col not in original_columns]
                                st.session_state.cat_cols = [
                                    col for col in cat_cols if col not in selected_encode_cols
                                ] + new_columns

                                st.success("One-Hot Encoding applied.")
                                st.dataframe(st.session_state.df_copy.head())
                            except Exception as e:
                                st.warning(f"Encoding Error: {str(e)}")
                    except Exception as e:
                        st.warning(f"Encoding Error: {str(e)}")
                else:
                    st.warning("Please select at least one column to encode.")
        else:
            st.info("No categorical columns to encode.")
    st.markdown("---")
    
# --- Train-Test Split ---
    
    st.header("Train & Test Split")
    with st.expander("Train-Test Split"):
        st.subheader("Split Your Dataset")
        test_size = st.slider("Choose test set size (%)", min_value=10, max_value=50, value=20, step=5)
        target_col = st.selectbox("Select target column", options=st.session_state.df_copy.columns)

        if st.button("Split Dataset"):
            try:
                X = st.session_state.df_copy.drop(columns=[target_col])
                y = st.session_state.df_copy[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)
                st.success(f"Dataset split completed ({100 - test_size}% train / {test_size}% test)")

                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test

                st.subheader("X_train Preview")
                st.dataframe(X_train.head())
                
                st.subheader("X_test Preview")
                st.dataframe(X_test.head())
                
                st.subheader("y_train Preview")
                st.dataframe(y_train.head())
                
                st.subheader("y_test Preview")
                st.dataframe(y_test.head())


            except Exception as e:
                st.warning(f"Error during split: {str(e)}")

#--- Model comparision---
    try:
        from xgboost import XGBClassifier, XGBRegressor
        xgboost_available = True
    except ImportError:
        xgboost_available = False

    st.markdown("---")
    st.subheader("Compare Multiple Models")

    # Step 1: Ensure Train-Test data is available
    if "X_train" not in st.session_state or "X_test" not in st.session_state:
        st.warning("Please perform Train-Test Split before comparing models.")
        st.stop()

    X_train = st.session_state["X_train"]
    X_test = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_test = st.session_state["y_test"]

    # Step 2: Let user select problem type here
    model_type = st.radio(
        "Select the type of problem you're solving:",
        ["Classification", "Regression"],
        key="compare_model_type"
    )

    # Save it globally for use in later steps (like model selection)
    st.session_state["model_type"] = model_type

    # Step 3: Define available models
    classification_models = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine (SVM)": SVC(),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(),
    }
    regression_models = {
        "Linear Regression": LinearRegression(),
        "KNN Regressor": KNeighborsRegressor(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Support Vector Regressor (SVR)": SVR(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Gradient Boosting Regressor": GradientBoostingRegressor(),
    }
    if xgboost_available:
        classification_models["XGBoost"] = XGBClassifier()
        regression_models["XGBoost Regressor"] = XGBRegressor()

    available_models = classification_models if model_type == "Classification" else regression_models

    # Step 4: Select models to compare
    selected_model_names = st.multiselect(
        "Select models to compare:",
        options=list(available_models.keys()),
        help="Select at least 2 models to compare."
    )

    # Step 5: Compare on button click
    if selected_model_names:
        if st.button("Compare Models"):
            results = []
            for name in selected_model_names:
                model = available_models[name]
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    if model_type == "Classification":
                        accuracy = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        results.append({"Model": name, "Accuracy": accuracy, "F1-Score": f1})
                    else:
                        mae = mean_absolute_error(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        results.append({"Model": name, "MAE": mae, "MSE": mse, "R²": r2})

                except Exception as e:
                    st.error(f"{name} failed to train or predict. Error: {e}")

            # Step 6: Show results
            if results:
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)

                # Plot and highlight best
                if model_type == "Classification":
                    best_model = results_df.loc[results_df["Accuracy"].idxmax()]
                    fig = px.bar(
                        results_df,
                        x="Model",
                        y=["Accuracy", "F1-Score"],
                        barmode="group",
                        title=f"Classification Model Comparison (Best: {best_model['Model']})",
                        color_discrete_sequence=["#636EFA", "#EF553B"]
                    )
                else:
                    best_model = results_df.loc[results_df["R²"].idxmax()]
                    fig = px.bar(
                        results_df,
                        x="Model",
                        y=["MAE", "MSE", "R²"],
                        barmode="group",
                        title=f"Regression Model Comparison (Best: {best_model['Model']})",
                        color_discrete_sequence=["#00CC96", "#FFA15A", "#AB63FA"]
                    )

                st.plotly_chart(fig, use_container_width=True)
                st.success(f"Best Performing Model:  `{best_model['Model']}`")

                # Step 7: Download CSV report
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Comparison Report",
                    data=csv,
                    file_name="model_comparison_report.csv",
                    mime="text/csv"
                )
    else:
        st.info("Please select at least two models to compare.")

    