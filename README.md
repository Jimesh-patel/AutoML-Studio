# AutoML-Studio

A Streamlit-based Machine Learning dashboard that allows users to **analyze, preprocess, train, compare, and evaluate models** without writing code.

---
### Try Live Demo : [AutoML-Studio](https://automl-studio.streamlit.app/)

## 📌 Features

* **Upload Dataset**: Upload CSV files and preview data with shape, summary, and statistics.

* **Exploratory Data Analysis (EDA)**:

  * Histogram, Boxplot, and Bar charts for columns
  * Correlation matrix for numerical data
  * Missing values heatmap

* **Data Preprocessing**:

  * Drop unwanted columns
  * Handle missing data (Mean, Median, Most Frequent, KNN, Constant)
  * Scale numerical features (StandardScaler, MinMaxScaler, RobustScaler)
  * Encode categorical features (Label Encoding, One-Hot Encoding)

* **Train-Test Split**: Split dataset into training and testing sets with custom ratios.

* **Model Comparison**:

  * Classification: Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes, Gradient Boosting, XGBoost*
  * Regression: Linear Regression, Ridge, Lasso, KNN Regressor, Decision Tree, Random Forest, SVR, Gradient Boosting, XGBoost*
  * Compare performance metrics (Accuracy, F1 for classification | MAE, MSE, R² for regression).

* **Model Building**:

  * Manual and automatic hyperparameter tuning (GridSearchCV).
  * Train and save selected models.

* **Model Evaluation**:

  * Classification: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC Curve
  * Regression: MAE, MSE, RMSE, R², Residual plots
  * Download trained model (`.pkl`) and evaluation report.

* **Prediction Interface**:

  * Upload new data for batch prediction
  * Manual input for single prediction
  * Probability outputs for classification models

---

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Jimesh-patel/AutoML-Studio.git
   cd automl-studio
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

Then open the link shown in the terminal (default: `http://localhost:8501`) in your browser.

---

## 📊 Example Workflow

1. Upload your dataset (`.csv`).
2. Explore the data (EDA section).
3. Preprocess: handle missing values, encode categorical, scale features.
4. Split dataset into train/test sets.
5. Compare models and pick the best one.
6. Train the final model (with hyperparameter tuning if needed).
7. Evaluate performance and download trained model.
8. Make predictions using new data.

---

## ✅ Requirements

Save the following block as `requirements.txt` in your project root.

```
streamlit>=1.0
pandas
numpy
seaborn
matplotlib
plotly
scikit-learn
xgboost
```

> Note: `xgboost` is optional — the app will work without it but will disable XGBoost-related features if it's not installed.

Install all with:

```bash
pip install -r requirements.txt
```

---

## 📌 Future Improvements

* Add deep learning models (TensorFlow / PyTorch).
* Support for time-series forecasting.
* Export preprocessing pipeline with the trained model.
* Auto-detect classification/regression tasks more robustly.

