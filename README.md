# Credit Risk Prediction

## Project Overview

This project is focused on predicting the credit risk associated with loan applications. The goal is to analyze the provided dataset, engineer relevant features, and build a robust classification model to predict whether a loan application is at risk (default) or not.

## Dataset Information

The dataset used in this project contains information about various loan applications, including features like the applicant's income, loan amount, credit history, and more. The primary aim is to predict the likelihood of default based on these features.

**Dataset Characteristics:**

- **Type:** Multivariate
- **Subject Area:** Finance
- **Associated Tasks:** Classification
- **Feature Type:** Categorical, Integer, Float
- **Number of Instances:** Variable based on the dataset

The dataset file used is **credit_risk_dataset.csv**.

## Project Objective

The objective of this project is to develop a classification model that can accurately predict the credit risk of loan applications. This involves extensive data analysis and feature engineering to improve model performance and provide actionable insights to financial institutions.

## Project Workflow

1. **Data Preprocessing:**
   - Handle missing values.
   - Encode categorical variables.
   - Scale and normalize features.

2. **Exploratory Data Analysis (EDA):**
   - Analyze the distribution of variables.
   - Visualize relationships between features.
   - Identify important features for prediction.

3. **Feature Engineering:**
   - Create new features to capture additional information.
   - Perform feature selection to retain the most relevant features.

4. **Model Development:**
   - Train various classification algorithms, including Logistic Regression, Decision Trees, Random Forest, ANN, and XGBoost.
   - Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.
   - Fine-tune the models to improve accuracy.

5. **Model Deployment:**
   - Deploy the final model (XGBoost) using Streamlit for a user-friendly interface.
   - Dockerize the application for containerization.
   - Host the application on AWS EC2 for real-time insights.

6. **CI/CD Pipeline:**
   - Automate deployment using CI/CD pipelines with GitHub Actions to ensure continuous integration and delivery.

## Tools and Technologies Used

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Model Deployment:** Streamlit, Docker, AWS EC2
- **CI/CD:** GitHub Actions
- **Version Control:** GitHub

## Results

The XGBoost model achieved an accuracy of 87%, making it the best-performing model among the trained algorithms. The solution was deployed on Streamlit and Docker, leveraging an AWS EC2 instance to provide real-time predictions. Continuous integration and delivery were ensured through automated CI/CD pipelines using GitHub Actions.

## Repository Structure

```plaintext
├── .github
│   └── workflows
│       └── ci.yml and cd.yml
├── .ipynb_checkpoints
├── ANN_model_pred.txt
├── Credit Risk Prediction.ipynb
├── Decision_tree_pred.txt
├── Dockerfile
├── Random_forest.txt
├── credit_risk_dataset.csv
├── creditapp.py
├── creidtdata_description.py
├── desktop.ini
├── feature_dict.json
├── logistic_model_pred.txt
├── requirements.txt
├── xgboost_model.joblib
└── xgboost_model_pred.txt
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/abhilitcode/credit-risk-prediction.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run creditapp.py
   ```
