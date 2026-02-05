from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import joblib

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.25,
    random_state=42,
    stratify=data.target
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "iris.joblib")

Y_test = model.predict(X_test)
print(classification_report(y_test, Y_test))
print(accuracy_score(y_test, Y_test))
print(confusion_matrix(y_test, Y_test))


