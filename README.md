# Credit Card Fraud Detection Pipeline

**Project Description:** Machine learning pipeline to detect and classify rare credit card fraud instances in a highly imbalanced dataset with a 0.172% fraud rate.


**#### PLEASE NOTE ####**
"This dataset is publicly available on Kaggle (or the original source). To run the project, please download the creditcard.csv file from the source and place it into a directory named data/ in the root of this project."

---

## 1. Project Goals and Challenges

This project implements an end-to-end data science solution aimed at accurately flagging fraudulent credit card transactions.

### Primary Goal
To deploy a model that achieves high **Recall** (minimizing missed fraud, which is critical for loss prevention) while maintaining an acceptable level of **Precision**.

### Key Challenges
* **Extreme Class Imbalance:** The positive class (fraud) accounts for only **0.172%** of all transactions, requiring specialized handling techniques.
* **Anonymized Data:** Most features (V1-V28) are principal components, requiring models to learn complex relationships without standard domain context.

---

## 2. Data Overview

The dataset consists of 284,807 transactions made by European cardholders in September 2013, over a two-day period.

| Metric | Value | Note |
| :--- | :--- | :--- |
| **Total Transactions** | 284,807 | |
| **Fraudulent Transactions** | 492 | The positive class (Target=1) |
| **Fraud Rate** | 0.172% | Defines the severe class imbalance |
| **Features** | `Time`, `Amount`, and 28 PCA-transformed features (V1-V28) | All features are numeric. |

---

## 3. Methodology

The methodology was designed specifically to address the imbalance problem and optimize for detection capability.

### Data Preprocessing
* **Feature Scaling:** The `Time` and `Amount` features were standardized.
* **[Specify Imbalance Technique, e.g., Undersampling / SMOTE / Class Weighting]**: **[Explain why you chose it, e.g., to reduce training time or introduce synthetic data.]**

### Model Selection
The final predictions were made using a **XGBoost Classifier** due to its robustness and performance on structured, imbalanced data.

---

## 4. 📈 Model Performance and Results

Model evaluation was prioritized using metrics suitable for imbalanced classification.

### Final Metrics on Test Set

**Metric**                          **Value**              **Importance for Fraud Detection**
Area Under the ROC Curve (AUC-ROC)    0.98                 High score indicating excellent separability between classes.
Recall (Sensitivity)                  84% (0.84)           CRITICAL: Measures the proportion of actual fraud cases correctly identified. 84% means only 16% of frauds were missed.
Precision                             76% (0.76)           Measures the proportion of flagged transactions that were truly fraud. 76% is a good trade-off.
F1-Score                              80% (0.80)           Harmonic mean of Precision and Recall for Class 1.
Accuracy                              100% (Overall)       High due to imbalance; not a reliable primary metric here.

**Conclusion:** The model achieved a high Recall score, demonstrating a strong capability to capture rare fraud events, thus minimizing potential financial loss.

---

## 5. Repository Structure and Dependencies

This project was developed using Python scripts in a VS Code environment.

### Project Files
* `src/`: Contains all custom Python scripts.
    * `src/credit_card.py`:  Data loading, feature engineering, and imbalance handling, Model fitting, hyperparameter tuning, and evaluation logic.
* `requirements.txt`: Lists all necessary dependencies.
* `README.md`: This document.

### Required Libraries
To replicate the environment, the following major libraries are needed:
* `pandas`
* `numpy`
* `scikit-learn`
* `xgboost`

---

## 6. Getting Started

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Wayne0792/Fraud-Detection-DataScience.git](https://github.com/Wayne0792/Fraud-Detection-DataScience.git)
    cd Fraud-Detection-DataScience
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the analysis:**
    * Execute the main script: `python src/model_train.py`

---

## 7. Contact

* **Author:** Wayne Sithole
* **GitHub:** [@Wayne0792](https://github.com/Wayne0792)
* **LinkedIn:** [www.linkedin.com/in/wayne-sithole-23b442202]

---

