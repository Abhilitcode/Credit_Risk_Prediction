import pandas as pd
import json

df = pd.read_csv('credit_risk_dataset.csv')
# df
# print(df.columns)

X = df[['person_age', 'person_income', 'person_home_ownership',
       'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income',
       'cb_person_default_on_file', 'cb_person_cred_hist_length']]

y = df['loan_status']

X.select_dtypes(include='object')

X_dummies = pd.get_dummies(X)
print(X_dummies.columns)

numerical_data = X.select_dtypes(include='number')
numerical_features = numerical_data.columns.tolist()

categorical_data = X.select_dtypes(exclude='number')
categorical_features = categorical_data.columns.tolist()

#feature dict will look like
# {
#     'person_home_ownership': ['RENT', 'OWN', 'MORTGAGE'],
#     'loan_intent': ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT']
# }
feature_dict = {}
for i in categorical_features:
    unique_vals = list(X[i].unique())
    feature_dict[i] = unique_vals 

with open('feature_dict.json','w') as fp:
    json.dump(feature_dict,fp)


