import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import roc_curve, auc
import pandas as pd 
import joblib
import json 

st.title('**Credit Risk Prediction in Banking Industry**')

with open('feature_dict.json', 'r') as fp:
    feature_dict = json.load(fp)

option = st.sidebar.selectbox(
    'Select a section:',
    ['Model Prediction', 'Data Vizualization']
)

if option == 'Model Prediction':
    with st.expander("**Project Overview**"):
        st.write("""
        **Project Objective:**

        The goal of this project is to create a prediction model that evaluates the credit risk associated with loan applications. We want to anticipate loan default properly by analyzing a large dataset of prior loan applications. This will enable financial institutions to make more informed lending decisions, ultimately improving risk management and lowering financial losses.

        **Project Name:**

        Credit Risk Prediction

        **Dataset Description:**

        **Credit Risk Dataset Overview:**

        The Credit Risk Dataset provides detailed information about various loan applications, including both financial and personal attributes of the applicants. This dataset is crucial for building predictive models to evaluate credit risk effectively.

        **Features:**

        1. **person_age**: Age of the applicant (numeric).
        2. **person_income**: Annual income of the applicant (numeric).
        3. **person_home_ownership**: Type of home ownership, such as RENT, OWN, MORTGAGE, or OTHER (categorical).
        4. **person_emp_length**: Employment length in years (numeric).
        5. **loan_intent**: Purpose of the loan, such as EDUCATION, MEDICAL, VENTURE, PERSONAL, HOME IMPROVEMENT, or DEBT CONSOLIDATION (categorical).
        6. **loan_grade**: Credit grade of the applicant, ranging from Excellent (A) to Unknown (G) (categorical).
        7. **loan_amnt**: Amount of the loan requested (numeric).
        8. **loan_intrate**: Interest rate on the loan (numeric).
        9. **loan_status**: Whether the loan defaulted or not (0 for non-default, 1 for default) (binary).
        10. **loan_percent_income**: Percentage of income allocated to the loan (numeric).
        11. **cb_person_default_on_file**: Historical default status (binary, Y for default, N for no default).
        12. **cb_person_cred_hist_length**: Length of credit history in years (numeric).
        """)


    #1st
    age = st.number_input('**Age**',min_value=0,max_value=145,step=1,format='%d')

    #2nd
    income = st.number_input('**Select Income**', min_value=0,max_value=2039785,format='%d')

    #3rd
    ownership = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
    home_ownership = st.selectbox('**Choose Home Ownership**', ownership)

    #4th
    person_emp_length = st.slider('**Employement Length (in years)**', min_value=0, max_value= 40, format='%d')

    #5th
    loan_intent_mapping = {
        "EDUCATION": "EDUCATION",
        "MEDICAL": "MEDICAL",
        "VENTURE": "VENTURE",
        "PERSONAL": "PERSONAL",
        "HOME IMPROVEMENT": "HOMEIMPROVEMENT",
        "DEBT CONSOLIDATION": "DEBTCONSOLIDATION",
    }

    loan_intent = st.selectbox('**Select Loan Intent**', list(loan_intent_mapping.keys()))

    #6th
    grade_mapping = {
        'Grade D - Satisfactory': 'D',
        'Grade B - Good': 'B',
        'Grade C - Fair': 'C',
        'Grade A - Excellent': 'A',
        'Grade E - Acceptable': 'E',
        'Grade F - Marginal': 'F',
        'Grade G - Critical': 'G'
    }

    loan_grade = st.selectbox('**Select Loan Grades (A: Low Risk to G: Higher Risk)**', list(grade_mapping.keys()))

    #7th
    loan_amount = st.number_input('**Loan Amount**', min_value=500, max_value=35000, format='%d')

    #8th
    loan_interest_rate = st.number_input('**Loan Interest Rate**', min_value=0.0, max_value=25.0)

    #9th
    loan_percentage_income = st.number_input('**Loan Percentage Of Income (E.g. 0.03 or 0.83)**', min_value=0.0, max_value=0.90)

    #10th
    status_mapping = {
        'Yes': 'Y',
        'No': 'N'
    }
    credit_file_status = st.radio('**Has Default on Credit File?**', list(status_mapping.keys()))

    #11th
    credit_hist = st.number_input('**Credit history (in years)**', min_value=0, max_value=40,format='%d')

    #12
    if st.button('**Predict Credit Risk**'):
        input_data = pd.DataFrame({
            'person_age': [age],
            'person_income': [income],
            'person_home_ownership': [home_ownership],
            'person_emp_length': [person_emp_length],
            'loan_intent': [loan_intent],
            'loan_grade': [loan_grade],
            'loan_amnt': [loan_amount],
            'loan_int_rate': [loan_interest_rate],
            'loan_percent_income': [loan_percentage_income],
            'cb_person_default_on_file': [credit_file_status],
            'cb_person_cred_hist_length': [credit_hist]
        })
        
        numerical_data = input_data.select_dtypes(include='number')
        numerical_features = numerical_data.columns.to_list()
        
        categorical_data = input_data.select_dtypes(exclude='number')
        categorical_features = categorical_data.columns.tolist()        
       
        X_test = {}
        for column in numerical_features:
            X_test[column] = input_data[column].iloc[0]
        
        for i in feature_dict:
            for j in feature_dict[i]:
                if input_data[i].iloc[0] == j:
                    X_test[f'{i}_{j}'] = [1]
                else:
                    X_test[f'{i}_{j}'] = [0]
        
        X_test_df = pd.DataFrame.from_dict(X_test)
        # print(X_test_df.columns)
        # print(X_test)
        
        expected_order = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
       'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER', 
       'person_home_ownership_OWN', 'person_home_ownership_RENT','loan_intent_DEBTCONSOLIDATION',
       'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT','loan_intent_MEDICAL', 'loan_intent_PERSONAL', 
       'loan_intent_VENTURE', 'loan_grade_A', 'loan_grade_B',
       'loan_grade_C', 'loan_grade_D', 'loan_grade_E', 'loan_grade_F',
       'loan_grade_G', 'cb_person_default_on_file_N',
       'cb_person_default_on_file_Y']
        
        X_test_df = X_test_df[expected_order]
        
        best_model = joblib.load('xgboost_model.joblib')
        
        predictions = best_model.predict(X_test_df)
        if predictions[0] == 0:
            st.write('Non Default')
        else:
            st.write("Default")
    
            
            
    
    
if(option=='Data Vizualization'):
    
    st.header('Data Vizualization')
    
    st.title("Top 15 Most Important Features")
    
    best_model = joblib.load('xgboost_model.joblib')
    
    xgb_model = best_model.best_estimator_
    
    feature_importance = pd.DataFrame({
        'Features': xgb_model.feature_names_in_,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Plot the top 15 features
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Features', data=feature_importance.head(15))
    plt.title('Top 15 Most Important Features (XGBoost)')
    st.pyplot(plt)
    plt.clf()
    
    
    df = pd.read_csv('credit_risk_dataset.csv')

    df = df[df["person_age"]<70]
    df = df[df['person_emp_length'] != 123]
    
    numerical_data = df.select_dtypes(include='number')
    numerical_features = numerical_data.columns.to_list()
    
    categorical_data = df.select_dtypes(exclude='number')
    categorical_features = categorical_data.columns.to_list()
    
    st.title("Correlation Matrix")
    plt.figure(figsize=(10,5))
    plt.title('Correlation Matrix of Numerical Features')
    sns.heatmap(data=numerical_data.corr(),annot=True,cmap='coolwarm',linewidths=0.5,fmt='.2f')
    st.pyplot(plt)
    plt.clf()
    
    st.title('Categorical Features Distribution by Loan Status')
    for category in categorical_features:
        plt.figure(figsize=(10, 5))
        sns.countplot(x=categorical_data[category], hue=numerical_data['loan_status'])
        plt.xticks(rotation=45)
        plt.title(f'Countplot of {category} with Loan Status')
        plt.xlabel(f'{category}')
        plt.ylabel('Count')
        st.pyplot(plt)
        plt.clf()
    
