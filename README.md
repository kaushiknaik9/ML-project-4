Customer Churn Prediction – Threshold Tuning

This project focuses on predicting customer churn using the IBM Telco Customer Churn Dataset and performing threshold tuning to optimize Precision, Recall, and F1 score. The goal is to build a reliable classification pipeline and analyze how different probability thresholds impact model performance.

Objective

To build a complete machine learning workflow that:

Predicts whether a customer will churn (Yes/No).

Tunes the classification threshold for maximizing Precision, Recall, or F1 Score depending on the objective.

Generates useful visualizations to support performance analysis.

Dataset

Source: IBM Telco Customer Churn Dataset
Format: CSV file

The dataset includes:

Demographic details

Services subscribed

Contract information

Billing and payment details

Target variable: Churn

Project Workflow
1. Data Exploration

Summary statistics

Distribution plots

Churn frequency

Identification of missing or incorrect values

Relationships between key variables (tenure, contract type, payment method, etc.)

2. Preprocessing

Key steps include:

Handling missing values

Converting TotalCharges to numeric

Encoding categorical features (binary + one-hot)

Scaling numerical columns (if required)

Splitting into train/test sets (80/20)

3. Model Development

A Logistic Regression model is trained as the baseline classifier.

Tasks completed:

Training using X_train, y_train

Generating probability scores using predict_proba

Initial evaluation using default threshold (0.5)

4. Threshold Tuning

Threshold tuning is performed to understand how decision boundaries affect the model.

Metrics measured across thresholds:

Precision

Recall

F1 Score

Performance curves and confusion matrices were generated for different thresholds.

5. Selecting the Optimal Threshold

Based on project requirements, the threshold that provides the best performance–for Precision, Recall, or F1–is identified.

This allows flexible decision-making:

Lower threshold → higher recall (catch more churn customers)

Higher threshold → higher precision (fewer false alarms)

Results

All threshold tuning plots are available in the /plots directory:

Precision vs Threshold

Recall vs Threshold

F1 Score vs Threshold

Confusion matrices for chosen thresholds

These plots help visualize trade-offs and guide the selection of the final threshold.
Repository Structure
/notebooks      → Jupyter notebooks for EDA, preprocessing, modeling  
/src            → Python scripts for model pipeline  
/plots          → Threshold tuning graphs and performance visuals  
/data           → Dataset (CSV)  
/reports        → Project report and documentation  
requirements.txt
run.sh
README.md

