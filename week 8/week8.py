import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Data Preprocessing
def preprocess_data(data):
    # Fill missing values
    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
    data['Married'].fillna(data['Married'].mode()[0], inplace=True)
    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
    data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
    data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    for column in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']:
        data[column] = label_encoder.fit_transform(data[column])
    
    return data

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Feature and Target variable
X = train_data.drop(columns=['Loan_ID', 'Loan_Status'])
y = train_data['Loan_Status']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Validation
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy}')

# Prediction on Test Data
test_data_X = test_data.drop(columns=['Loan_ID'])
test_data['Loan_Status'] = model.predict(test_data_X)

# Prepare Submission
submission = test_data[['Loan_ID', 'Loan_Status']]
submission.to_csv('submission.csv', index=False)
