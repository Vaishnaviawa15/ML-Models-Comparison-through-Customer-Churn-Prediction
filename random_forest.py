<<<<<<< HEAD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc

# Load the dataset
data = pd.read_csv('D:\Projects\ML Project\ML-Models-Comparison-Through-Customer-Churn-Prediction\Cleaned_E_Commerce_Data.csv')

# Encode categorical variables
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features and target
target = 'Churn'  # Assuming 'Churn' is the prediction target
X = data.drop(columns=[target])
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# Evaluate Model
def evaluate_model(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"{model_name} Performance:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print(classification_report(y_true, y_pred))
    print("\n" + "-" * 50 + "\n")

    return [accuracy, precision, recall, f1]

# Store results for visualization
metrics = {}
metrics["Random Forest"] = evaluate_model("Random Forest", y_test, rf_predictions)

# Convert results into a DataFrame for visualization
metrics_df = pd.DataFrame(metrics, index=["Accuracy", "Precision", "Recall", "F1 Score"])

# Plot the metrics
plt.figure(figsize=(10, 5))
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title("Random Forest Model Performance")
plt.ylabel("Score")
plt.xlabel("Metrics")
plt.xticks(rotation=0)
plt.legend(title="Model")
plt.show()

# Plot ROC Curve
plt.figure(figsize=(10, 6))
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
auc_rf = auc(fpr_rf, tpr_rf)

plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.show()
=======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc

# Load the dataset
data = pd.read_csv('D:\Projects\ML Project\ML-Models-Comparison-Through-Customer-Churn-Prediction\Cleaned_E_Commerce_Data.csv')

# Encode categorical variables
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features and target
target = 'Churn'  # Assuming 'Churn' is the prediction target
X = data.drop(columns=[target])
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# Evaluate Model
def evaluate_model(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"{model_name} Performance:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print(classification_report(y_true, y_pred))
    print("\n" + "-" * 50 + "\n")

    return [accuracy, precision, recall, f1]

# Store results for visualization
metrics = {}
metrics["Random Forest"] = evaluate_model("Random Forest", y_test, rf_predictions)

# Convert results into a DataFrame for visualization
metrics_df = pd.DataFrame(metrics, index=["Accuracy", "Precision", "Recall", "F1 Score"])

# Plot the metrics
plt.figure(figsize=(10, 5))
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title("Random Forest Model Performance")
plt.ylabel("Score")
plt.xlabel("Metrics")
plt.xticks(rotation=0)
plt.legend(title="Model")
plt.show()

# Plot ROC Curve
plt.figure(figsize=(10, 6))
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
auc_rf = auc(fpr_rf, tpr_rf)

plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.show()
>>>>>>> 6beea00e2bacdf6a04b11b2a917af4c0134bb444
