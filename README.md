

---


# 🏡 Forecasting House Prices Using Smart Regression Techniques

## 📌 Problem Statement
The real estate market is dynamic and influenced by various factors—location, size, age, and economic conditions. Traditional price estimation methods fail to capture the non-linear complexities.  
This project uses **machine learning regression models** to accurately forecast house prices, helping buyers, sellers, and investors make data-driven decisions.

---

## 🎯 Objectives

- 🧹 Collect and clean real estate data for modeling
- 📊 Perform Exploratory Data Analysis (EDA)
- 🧠 Build and compare multiple regression models
- 🥇 Identify the most accurate prediction model
- 📈 Visualize results and communicate insights
- 🌐 *(Optional)* Deploy an interactive tool

---

## 🔍 Scope

### ✅ Inclusions:
- Supervised regression techniques
- Feature engineering & performance evaluation
- Visualization of trends and model outputs

### ⚠️ Limitations:
- Works on structured, numeric data only
- Uses static/historical datasets
- Deployment is optional and limited

---

## 🗂️ Dataset Info

- **Dataset**: Boston Housing Dataset  
- **Source**: `sklearn.datasets`  
- **Format**: Numeric features, with median home value (`medv`) as target

---

## 🔧 High-Level Methodology

### 1. 📥 Data Collection
- Used Boston Housing dataset from `sklearn.datasets`

### 2. 🧹 Data Cleaning
- Checked for nulls, data types, and duplicates using:
  ```python
  print(df.info())
  print(df.isnull().sum())
  ```

### 3. 📊 Exploratory Data Analysis (EDA)
- Distribution and correlation analysis:
  ```python
  sns.histplot(df['medv'], kde=True)
  plt.title('Distribution of House Prices')
  plt.show()

  plt.figure(figsize=(10,8))
  sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
  plt.title('Correlation Heatmap')
  plt.show()
  ```

### 4. 🧰 Feature Engineering
- No new features added
- Polynomial terms optionally considered

### 5. 🤖 Model Building

Models trained and evaluated:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- XGBoost Regressor

#### Example code:
```python
# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
results['Linear Regression'] = r2_score(y_test, y_pred_lr)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
results['Ridge Regression'] = r2_score(y_test, y_pred_ridge)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
results['Lasso Regression'] = r2_score(y_test, y_pred_lasso)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = r2_score(y_test, y_pred_rf)

# XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
results['XGBoost'] = r2_score(y_test, y_pred_xgb)
```

---

### 6. 📈 Model Evaluation

Metrics Used:
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

```python
for model, score in results.items():
    print(f"{model}: R² Score = {score:.4f}")

print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
```

---

### 7. 📉 Visualization & Interpretation

```python
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='purple')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices - Random Forest")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
plt.show()
```

---

### 8. 🚀 Deployment *(Optional)*

- Use **Streamlit** or **Jupyter Notebook** for interactive visualization

---

## 🧰 Tools & Technologies

| Category | Tools |
|----------|-------|
| Programming Language | Python |
| Notebook / IDE | Google Colab |
| Libraries | `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost` |
| Deployment (Optional) | Streamlit / Flask |

---


