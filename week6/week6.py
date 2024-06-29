import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('titanic.csv')

# Display the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Fill missing Age values with the median age
data['Age'].fillna(data['Age'].median(), inplace=True)

# Fill missing Embarked values with the most common port
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop the Cabin column as it has too many missing values
data.drop('Cabin', axis=1, inplace=True)

# Create a new feature 'FamilySize'
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# Create a new feature 'IsAlone'
data['IsAlone'] = 1  # Initialize to 1 (True)
data['IsAlone'].loc[data['FamilySize'] > 1] = 0  # Recode to 0 (False)

# Extract Title from Name
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Simplify titles
data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                       'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace('Mlle', 'Miss')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')

# Map titles to numerical values
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
data['Title'] = data['Title'].map(title_mapping)
data['Title'] = data['Title'].fillna(0)

# Encode 'Sex'
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])

# Encode 'Embarked'
data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])

# Select features for scaling
features_to_scale = ['Age', 'Fare', 'FamilySize']

# Initialize the scaler
scaler = StandardScaler()

# Scale selected features
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Drop unnecessary columns
data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# Separate features and target variable
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the preprocessed data
print(X_train.head())
