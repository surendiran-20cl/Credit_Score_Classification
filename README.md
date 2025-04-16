# Credit Score Classification Using Supervised Machine Learning

## Overview

This project involves developing a predictive machine learning system to classify individuals into credit score brackets based on historical credit and financial attributes. The goal is to automate the classification process to support risk assessment and reduce manual efforts for financial institutions.

This is a **multiclass classification** problem with the target variable being `Credit_Score`, categorized into three classes:
- **0**: Poor
- **1**: Standard
- **2**: Good

The end-to-end pipeline includes preprocessing, feature engineering, exploratory data analysis (EDA), model training, hyperparameter tuning, performance evaluation, and interpretability via feature importance and ROC curves.



## Problem Statement

A global financial organization has collected extensive credit-related data of individuals. The task is to build an intelligent classification system that can predict a person's credit score category (`Poor`, `Standard`, or `Good`) using supervised learning techniques.



## Dataset Description

The dataset consists of **100,000 records** with **28 features**, including demographic data, credit behavior, financial transactions, and account history.

### Feature Description

| Feature                      | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `Age`                        | Age of the customer                                                         |
| `Occupation`                 | Job type or profession                                                      |
| `Annual_Income`              | Annual income in USD                                                        |
| `Monthly_Inhand_Salary`      | Monthly salary in hand                                                      |
| `Num_Bank_Accounts`          | Total bank accounts held                                                    |
| `Num_Credit_Card`            | Number of credit cards owned                                                |
| `Interest_Rate`              | Interest rate on current credit                                             |
| `Num_of_Loan`                | Number of loans held                                                        |
| `Type_of_Loan`               | Types of loans taken (personal, auto, mortgage, etc.)                       |
| `Delay_from_due_date`        | Average number of days late for credit payments                            |
| `Num_of_Delayed_Payment`     | Total number of delayed payments                                            |
| `Changed_Credit_Limit`       | % change in credit card limit                                               |
| `Num_Credit_Inquiries`       | Number of recent credit checks/inquiries                                    |
| `Credit_Mix`                 | Quality of the credit mix (Good, Standard, Bad)                             |
| `Outstanding_Debt`           | Remaining unpaid debt                                                       |
| `Credit_Utilization_Ratio`   | Credit used as a percentage of limit                                        |
| `Credit_History_Age`         | Duration of credit history in months                                        |
| `Payment_of_Min_Amount`      | Whether the minimum credit amount was paid                                  |
| `Total_EMI_per_month`        | Total EMI payments per month                                                |
| `Amount_Invested_monthly`    | Monthly amount invested                                                     |
| `Payment_Behaviour`          | Credit card payment behavior                                                |
| `Monthly_Balance`            | Net balance at the end of the month                                         |
| `Credit_Score` (Target)      | Credit score class (0=Poor, 1=Standard, 2=Good)                             |


## Tech Stack

- **Language**: Python 3.11+
- **Development Platform**: Google Colab
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`: model training, evaluation, preprocessing
  - `statsmodels`: multicollinearity (VIF)
  - `GridSearchCV`: hyperparameter optimization



## Project Workflow

### 1. Data Preprocessing
- Removal of irrelevant identifiers (e.g., `ID`, `Name`, `SSN`)
- Handling missing values using forward and backward fill
- Type conversions for numeric and temporal fields
- Outlier removal in `Age` via IQR filtering
- Text parsing of `Credit_History_Age` to numeric (months)

### 2. Feature Engineering
- One-hot encoding of categorical features (`Occupation`, `Type_of_Loan`, etc.)
- Label encoding of ordinal features (`Credit_Mix`, `Credit_Score`)
- Correlation analysis and multicollinearity checks using VIF

### 3. Exploratory Data Analysis (EDA)
- Distribution plots for income, age, credit score
- Boxplots comparing features across target classes
- Countplots for categorical influence
- Correlation heatmaps for numeric variables

### 4. Model Development
Four supervised learning models were trained and evaluated:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Tuned Decision Tree** using `GridSearchCV`
- **Random Forest Classifier**

All models were trained using `train_test_split` with an 80-20 ratio and evaluated on unseen data.

### 5. Evaluation Metrics
- **Accuracy**, **Precision**, **Recall**, **F1-Score** (macro & weighted)
- **Confusion Matrix** visualizations
- **Feature Importance** plots for interpretability
- **Multiclass ROC Curves** and **AUC** per class using One-vs-Rest classifiers


## Model Performance Summary

| Model               | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
|---------------------|----------|-------------------|----------------|------------------|
| Logistic Regression | 73.5%    | 0.71              | 0.71           | 0.71             |
| Decision Tree       | 73.2%    | 0.71              | 0.71           | 0.71             |
| Tuned Decision Tree | 72.6%    | 0.70              | 0.72           | 0.71             |
| Random Forest       | **80.2%**| **0.79**          | **0.79**       | **0.79**         |

**Conclusion**: The Random Forest Classifier outperformed all other models in terms of both overall accuracy and class-level consistency, making it the most robust model for deployment in this context.



## Visual Outputs

- Feature Importance Bar Charts (Logistic & Tree-based)
- Confusion Matrix Heatmaps
- Classification Report Heatmaps
- Multiclass ROC-AUC Curves (per model)




## Future Improvements

- Incorporate external credit behavior datasets for feature enrichment
- Experiment with gradient boosting algorithms (XGBoost, LightGBM)
- Apply cross-validation and SMOTE for class imbalance, if needed
- Deploy the model using Flask/FastAPI for production scenarios



## Author

This project is part of the **Artificial Intelligence Certification Program** from Intellipaat and was implemented entirely in Python using Google Colab.



## License

This project is for academic and educational purposes. Contact the author for commercial use or collaborations.
