# iris_classifier.py
# Project 2 - Data Classification Using AI
# DecodeLabs Industrial Training | Batch 2026
# Algorithm: K-Nearest Neighbors (KNN) on the Iris Dataset

# -- Imports -----------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


# -- Step 1: Load Dataset ------------------------------------------------
iris = load_iris()
X = iris.data      # features: sepal length, sepal width, petal length, petal width
y = iris.target     # labels: 0=setosa, 1=versicolor, 2=virginica

print('Dataset loaded successfully.')
print(f'Total samples : {X.shape[0]}')
print(f'Features      : {X.shape[1]}')
print(f'Classes       : {iris.target_names}')


# -- Step 2: Explore the Data --------------------------------------------
print('\nFeature names:', iris.feature_names)
print('First 5 rows:\n', X[:5])
print('Corresponding labels:', y[:5])

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
for cls, cnt in zip(iris.target_names, counts):
    print(f'  {cls}: {cnt} samples')


# -- Step 3: Feature Scaling (StandardScaler) -----------------------------
# KNN uses distance to find neighbours, so scaling is important.
# Without it, a feature with larger values dominates the distance calculation.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print('\nScaling done. First row before scaling:', X[0])
print('First row after scaling :', X_scaled[0].round(3))


# -- Step 4: Train-Test Split ---------------------------------------------
# 80% training, 20% testing
# stratify=y makes sure each class is proportionally represented in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f'\nTraining samples : {X_train.shape[0]}')
print(f'Testing samples  : {X_test.shape[0]}')


# -- Step 5: Train the KNN Model -------------------------------------------
# k=5 means the model looks at 5 nearest neighbours and takes a majority vote
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

print('\nModel trained successfully.')


# -- Step 6: Make Predictions ----------------------------------------------
y_pred = model.predict(X_test)


# -- Step 7: Evaluate the Model --------------------------------------------
acc = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {acc * 100:.2f}%')

print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print('Confusion Matrix:')
cm = confusion_matrix(y_test, y_pred)
print(cm)


# -- Step 8: Predict on New Custom Input -----------------------------------
# Example: a new flower with sepal_length=5.1, sepal_width=3.5,
#          petal_length=1.4, petal_width=0.2
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)

print(f'\nCustom Prediction:')
print(f'Input features   : {new_flower[0]}')
print(f'Predicted class  : {iris.target_names[prediction[0]]}')
