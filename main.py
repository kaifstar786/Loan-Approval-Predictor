
# Loan Approval Predictor
# Educational project using a dummy dataset

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create a sample dataset
data = {
    "income": [
        30000, 45000, 60000, 25000, 80000,
        55000, 35000, 90000, 40000, 70000,
        50000, 65000, 28000, 85000, 75000,
        32000, 58000, 95000, 42000, 68000
    ],
    "loan_amount": [
        15000, 20000, 25000, 18000, 30000,
        22000, 17000, 35000, 16000, 28000,
        21000, 24000, 19000, 32000, 27000,
        14000, 23000, 38000, 18000, 26000
    ],
    "credit_history": [
        1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
        0, 1, 0, 1, 1, 0, 1, 1, 0, 1
    ],
    "employment": [
        "Yes", "Yes", "Yes", "No", "Yes",
        "Yes", "No", "Yes", "Yes", "Yes",
        "No", "Yes", "No", "Yes", "Yes",
        "No", "Yes", "Yes", "No", "Yes"
    ],
    "approval": [
        1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
        0, 1, 0, 1, 1, 0, 1, 1, 0, 1
    ]
}

df = pd.DataFrame(data)

# Step 2: Save and display dataset
df.to_csv("loan_data.csv", index=False)

print("\n===== LOAN APPROVAL DATASET =====")
print(df.head(10))

# Step 3: Prepare input and output
X = df[["income", "loan_amount", "credit_history", "employment"]]
y = df["approval"]

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Step 5: Prepare preprocessing
numeric_features = ["income", "loan_amount", "credit_history"]
categorical_features = ["employment"]

preprocessor = ColumnTransformer([
    ("numbers", StandardScaler(), numeric_features),
    ("categories", OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
])

# Step 6: Create Machine Learning model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

# Step 7: Train model
model.fit(X_train, y_train)

# Step 8: Check model performance
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n===== MODEL PERFORMANCE =====")
print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Step 9: Visualize dataset
plt.figure(figsize=(7, 4))

df["approval"].value_counts().sort_index().plot(
    kind="bar", color=["tomato", "seagreen"]
)

plt.title("Loan Approval Distribution (Dummy Data)")
plt.xlabel("Approval Status (0 = No, 1 = Yes)")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("loan_approval_chart.png")
plt.show()

# Step 10: Make a sample prediction
sample = pd.DataFrame({
    "income": [60000],
    "loan_amount": [20000],
    "credit_history": [1],
    "employment": ["Yes"]
})

result = model.predict(sample)

print("\n===== SAMPLE PREDICTION =====")

if result[0] == 1:
    print("Model Prediction: Likely Approved")
else:
    print("Model Prediction: Likely Not Approved")

print("\nThis is an educational prediction using dummy data.")
print("It is not a real loan decision.")
