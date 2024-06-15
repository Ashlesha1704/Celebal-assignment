import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
data = pd.read_csv('Books.csv')

# Display the first few rows of the dataset
print(data.head())

# Define the preprocessing steps for numerical features
numerical_features = ['feature1', 'feature2']  # replace with actual numerical feature names
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define the preprocessing steps for categorical features
categorical_features = ['feature3', 'feature4']  # replace with actual categorical feature names
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Apply the preprocessing steps to the dataset
preprocessed_data = preprocessor.fit_transform(data)

# Convert the preprocessed data back to a DataFrame
preprocessed_df = pd.DataFrame(preprocessed_data)

# Display the preprocessed data
print(preprocessed_df.head())

# Feature engineering example: create new features
# Add new feature as a combination of existing features (example)
data['new_feature'] = data['feature1'] * data['feature2']  # replace with actual feature names

# Display the data with the new feature
print(data.head())

# Save the preprocessed and engineered data to a new CSV file
data.to_csv('books_preprocessed.csv', index=False)
