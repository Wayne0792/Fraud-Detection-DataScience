import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler


# =================================================================
## 1. DATA LOADING
# =================================================================
print("--- 1. Data Loading ---")
df = pd.read_csv('creditcard.csv')
print("DataFrame Head:")
print(df.head())

# =================================================================
## 2. CLASS IMBALANCE ANALYSIS
# =================================================================
print("\n--- 2. Class Imbalance Analysis ---")
class_counts = df['Class'].value_counts()
total_transactions = len(df)
print("\n### Class Distribution (Count):")
print(class_counts)
print("\n### Class Distribution (Percentage):")
print(f"Non-Fraud (0): {class_counts[0] / total_transactions * 100:.4f}%")
print(f"Fraud (1): {class_counts[1] / total_transactions * 100:.4f}%")


# =================================================================
## 3. EDA on TIME and AMOUNT
# =================================================================
print("\n--- 3. EDA on TIME and AMOUNT ---")

# 3.1. Analyze Transaction Time
df['Hour'] = df['Time'].apply(lambda x: np.ceil(x / 3600))
plt.figure(figsize=(10, 6))
sns.histplot(df[df['Class'] == 0]['Hour'], bins=48, kde=False, label='Non-Fraud', color='blue', alpha=0.6)
sns.histplot(df[df['Class'] == 1]['Hour'], bins=48, kde=False, label='Fraud', color='red', alpha=0.9)
plt.title('Transaction Count by Hour (0 to 48 hours)')
plt.xlabel('Time (Hours)')
plt.ylabel('Transaction Count')
plt.legend()
plt.show() # This command displays the plot

# 3.2. Analyze Transaction Amount
print("\n### Amount Statistics (Fraud vs. Non-Fraud):")
print("Fraudulent Transactions:")
print(df[df['Class'] == 1]['Amount'].describe())
print("\nNon-Fraudulent Transactions:")
print(df[df['Class'] == 0]['Amount'].describe())

plt.figure(figsize=(8, 5))
sns.boxplot(x='Class', y='Amount', data=df, showfliers=False)
plt.title('Transaction Amount Distribution by Class (No Extreme Outliers)')
plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
plt.ylabel('Transaction Amount')
plt.show() # This command displays the plot

# Clean up temporary column
df = df.drop('Hour', axis=1)

print("\n--- Initial Analysis Complete ---")


# Initialize the StandardScaler
scaler = StandardScaler()

# Scale 'Amount' and 'Time' features
df['Scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop the original 'Time' and 'Amount' columns, as we will use the scaled versions
df = df.drop(['Time', 'Amount'], axis=1)

# Display the head with the new scaled features to verify
print("DataFrame Head with Scaled Features:")
print(df.head())

# Separate features (X) and target (y)
X = df.drop('Class', axis=1) # All columns except 'Class'
y = df['Class']             # The 'Class' column


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # For reproducibility
    stratify=y           # ENSURES class balance is maintained in both sets
)

# Print the shape of the resulting sets
print("\nData Split Shapes:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


from sklearn.utils import class_weight
import numpy as np

# Calculate the required weight for the minority class (Class 1)
# This finds how many times more instances of Class 0 there are than Class 1.
scale_pos_weight = (len(y_train[y_train == 0]) / len(y_train[y_train == 1]))
print(f"Calculated Scale Position Weight (Majority / Minority): {scale_pos_weight:.2f}")

# The XGBoost model will use this value to heavily penalize False Negatives (missing a fraud case).


import xgboost as xgb

# Initialize the XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # For binary classification
    eval_metric='auc',            # We'll evaluate using AUC-ROC
    use_label_encoder=False,      # Required for newer XGBoost versions
    random_state=42,
    # Apply the weight to combat imbalance
    scale_pos_weight=scale_pos_weight,
    # A few parameters for speed/performance
    n_estimators=100,
    learning_rate=0.1
)

# Train the model
print("\nTraining XGBoost Classifier...")
xgb_model.fit(X_train, y_train)
print("Training Complete.")

# Predict probabilities (needed for AUC)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Predict the final class labels (0 or 1)
y_pred = xgb_model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 1. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("### Confusion Matrix:")
print(conf_matrix)

# 2. Classification Report (Contains Precision, Recall, and F1-Score)
print("\n### Classification Report:")
print(classification_report(y_test, y_pred))

# 3. AUC-ROC Score
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\n### AUC-ROC Score: {auc_score:.4f}")


# Plotting the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Baseline (AUC = 0.5)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR / Recall)')
plt.legend(loc='lower right')
plt.show()


# Get feature importance from the trained XGBoost model
importance = xgb_model.feature_importances_
feature_names = X_train.columns
sorted_idx = np.argsort(importance)[::-1] # Sort features by importance

# Plot the top 10 most important features
plt.figure(figsize=(10, 7))
sns.barplot(x=importance[sorted_idx[:10]], y=feature_names[sorted_idx[:10]])
plt.title('Top 10 Feature Importance')
plt.xlabel('F-Score (Importance)')
plt.show()