import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Image Preprocessing Function
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Feature Extraction Function
def extract_features(image):
    features = []
    # HOG features
    hog_features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    features.extend(hog_features)
    # LBP features
    lbp_features = local_binary_pattern(image, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp_features, bins=np.arange(0, 11), range=(0, 10))
    features.extend(lbp_hist)
    # Flattened pixel values (optional)
    features.extend(image.flatten())
    return np.array(features)

# Prepare training data
X = []
y = []
for idx, row in train_df.iterrows():
    image_path = os.path.join('images', f"{row['image_id']}.jpg")
    image = preprocess_image(image_path)
    features = extract_features(image)
    X.append(features)
    y.append(row['combinations'])  # Modify based on target labels (one-hot encode if needed)

X = np.array(X)
y = np.array(y)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Apply PCA for feature selection
pca = PCA(n_components=100)  # Adjust number of components based on explained variance
X = pca.fit_transform(X)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grids
svm_params = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10]
}

rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

gbm_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Initialize models
svm = SVC()
rf = RandomForestClassifier()
gbm = GradientBoostingClassifier()

# Perform hyperparameter tuning
svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='accuracy')
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy')
gbm_grid = GridSearchCV(gbm, gbm_params, cv=5, scoring='accuracy')

svm_grid.fit(X_train, y_train)
rf_grid.fit(X_train, y_train)
gbm_grid.fit(X_train, y_train)

# Print best parameters and scores
print("Best SVM Params:", svm_grid.best_params_)
print("Best RF Params:", rf_grid.best_params_)
print("Best GBM Params:", gbm_grid.best_params_)

# Validate models
svm_pred = svm_grid.predict(X_val)
rf_pred = rf_grid.predict(X_val)
gbm_pred = gbm_grid.predict(X_val)

print("SVM Accuracy:", accuracy_score(y_val, svm_pred))
print("RF Accuracy:", accuracy_score(y_val, rf_pred))
print("GBM Accuracy:", accuracy_score(y_val, gbm_pred))

# Ensemble Model
ensemble = VotingClassifier(estimators=[
    ('svm', svm_grid.best_estimator_),
    ('rf', rf_grid.best_estimator_),
    ('gbm', gbm_grid.best_estimator_)
], voting='hard')

ensemble.fit(X_train, y_train)
ensemble_pred = ensemble.predict(X_val)

print("Ensemble Accuracy:", accuracy_score(y_val, ensemble_pred))

# Prepare test data and generate submission
test_features = []
for idx, row in test_df.iterrows():
    image_path = os.path.join('images', f"{row['image_id']}.jpg")
    image = preprocess_image(image_path)
    features = extract_features(image)
    test_features.append(features)

test_features = np.array(test_features)

# Standardize and apply PCA to test features
test_features = scaler.transform(test_features)
test_features = pca.transform(test_features)

# Predict using the ensemble model
test_predictions = ensemble.predict(test_features)

# Create submission DataFrame
submission_df = pd.DataFrame({
    'image_id': test_df['image_id'],
    'combinations': test_predictions
})

submission_df.to_csv('submission.csv', index=False)
