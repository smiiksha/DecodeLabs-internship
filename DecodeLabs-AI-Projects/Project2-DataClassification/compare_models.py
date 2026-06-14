# compare_models.py
# Comparing KNN vs Decision Tree on the Iris Dataset
# Project 2 - Data Classification Using AI

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train both models
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test))

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))

print(f'KNN Accuracy           : {knn_acc * 100:.2f}%')
print(f'Decision Tree Accuracy : {dt_acc * 100:.2f}%')

if knn_acc >= dt_acc:
    print('KNN performed better or equally well on this dataset.')
else:
    print('Decision Tree performed better on this dataset.')
