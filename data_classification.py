from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Dataset Loaded Successfully")
print("Total Samples:", len(X))

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", len(X_train))
print("Testing Data:", len(X_test))

# Train Model
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

print("\nModel Training Completed")

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Sample Prediction
sample = [[5.1, 3.5, 1.4, 0.2]]

result = model.predict(sample)

print("\nPrediction Result:")
print("Flower Class:", iris.target_names[result[0]])