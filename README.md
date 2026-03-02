# Customer Churn Prediction.
A machine learning project focused on forecasting customer churn for an e-commerce platform using the E-Commerce Customer Churn Dataset. The study evaluates Logistic Regression, Random Forest, and XGBoost models, with XGBoost delivering the highest performance (Accuracy = 0.97, F1-Score = 0.89).

[Live Demo](https://gen-ai-capstone-project.streamlit.app/)
*(Hosted on Streamlit Cloud)*

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)

## Overview
This project performs an end-to-end machine learning pipeline to predict customer churn. It covers:
- **Data Loading & Exploration** — Understanding customer behavior and churn patterns.
- **Data Preprocessing** — Handling missing values, encoding categorical features, and feature scaling.
- **Model Training & Evaluation** — Comparing Logistic Regression, Random Forest, and XGBoost.
- **Model Deployment** — A Streamlit web application for real-time churn prediction.

## Dataset
| Property | Details |
| :--- | :--- |
| **File** | `E Commerce Dataset.xlsx` |
| **Rows** | 5,630 |
| **Columns** | 20 |
| **Target Variable** | `Churn` (1: Churned, 0: Retained) |

### Features
| Feature | Type | Description |
| :--- | :--- | :--- |
| **Tenure** | float64 | Months customer has been with the service |
| **CityTier** | int64 | City classification (Tier 1, 2, or 3) |
| **WarehouseToHome** | float64 | Distance from warehouse to home |
| **PreferredLoginDevice** | object | Device used for login (Mobile, Computer) |
| **PreferredPaymentMode** | object | Payment method (UPI, CC, Debit, etc.) |
| **Gender** | object | Male or Female |
| **HourSpendOnApp** | float64 | Average hours spent on app |
| **NumberOfDeviceRegistered**| int64 | Total devices linked to account |
| **SatisfactionScore** | int64 | Customer satisfaction rating (1-5) |
| **Complain** | int64 | Whether the customer filed a complaint (1/0) |
| **OrderCount** | float64 | Total number of orders placed |
| **DaySinceLastOrder** | float64 | Days since last purchase |
| **CashbackAmount** | float64 | Total cashback received |

## Project Workflow
1. **Data Loading & Exploration**
   - Load dataset from Excel using Pandas.
   - Analyze dataset shape, statistics, and missing values (`isnull().sum()`).
   - Visualize churn distribution and feature correlations.

2. **Data Preprocessing**
   - **Missing Value Handling**: Numerical features imputed with `mean`, categorical features with `most_frequent`.
   - **Feature Engineering**: Standardized skewed features and handled outliers.
   - **Label Encoding**: Categorical variables converted for model compatibility.
   - **Dropped Columns**: `CustomerID` (non-predictive feature).

3. **Train-Test Split**
   - 80% Training / 20% Testing with `random_state=42`.

4. **Model Training & Evaluation**
   - **Logistic Regression**: Baseline model with balanced class weights.
   - **Random Forest**: Ensemble model for non-linear patterns.
   - **XGBoost**: Gradient boosted trees for high-performance prediction.

5. **Model Selection**
   - Selected the model with the highest F1-Score and Accuracy for the final pipeline.

## Results
| Metric | Logistic Regression | Random Forest | XGBoost |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.90 | 0.96 | **0.97** |
| **Precision (Class 1)** | 0.76 | 0.93 | **0.92** |
| **Recall (Class 1)** | 0.56 | 0.80 | **0.87** |
| **F1-Score (Class 1)** | 0.65 | 0.86 | **0.89** |

**Best Model**: **XGBoost Classifier** — Selected for deployment due to superior recall and F1-score.

## Tech Stack
| Library | Purpose |
| :--- | :--- |
| **Python 3.11** | Programming language |
| **Pandas / NumPy** | Data manipulation |
| **Scikit-learn** | ML models and preprocessing |
| **XGBoost** | High-performance boosting model |
| **Streamlit** | Web application framework |
| **Matplotlib / Seaborn** | Data visualization |
| **Pickle** | Model Creation |

## Project Structure
```
gen_ai_project_churn/
├── E Commerce Dataset.xlsx   # Customer churn dataset
├── customer_churn.ipynb      # Main Jupyter Notebook (EDA & Modeling)
├── churn_pipeline.pkl        # Serialized best model (XGBoost)
├── streamlit_app.py          # Streamlit live demo script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Getting Started
### Prerequisites
- Python 3.8
- Jupyter Notebook or VS Code

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd gen_ai_project_churn
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

4. Launch the notebook:
   ```bash
   jupyter notebook customer_churn.ipynb
   ```

## Team Members
| Name | Enrollment No. |
| :--- | :--- |
| **Rudraksh Sharma** | 2401010395 |
| **Bineet Keshari** | 2401010130 |
| **Vridhi Chaudhary** | 2401010336 |
| **Anshuman Mehta** | 2401010082 |
