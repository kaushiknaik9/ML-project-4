# Telco Customer Churn Prediction

## Project Overview

This project predicts customer churn for a telecom company using the IBM Telco Customer Attrition dataset. The workflow includes data loading, preprocessing, model training using logistic regression, threshold tuning, and metric evaluation.

## Structure

- `data/`: Contains telco_churn.csv dataset
- `src/`: Contains Python modules (`data_loader.py`, `preprocess.py`, `train_model.py`, `metrics.py`)
- `notebooks/`: Jupyter notebooks for exploratory work
- `plots/`: Output plots of metrics
- `ML_Project_Report-1.pdf`: Project report
- `Threshold-Tuning-for-Telco-Customer-Churn-Prediction.pptx`: Project presentation

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the main notebook or script from `notebooks/`.
3. Inspect output plots and metrics (`plots/`).

## Main Steps

1. **Load Data**  
   Loads CSV data and cleans critical columns.

2. **Preprocess Data**

- Categorical encoding
- Numerical scaling
- Train/test splitting

3. **Train Model**  
   Uses logistic regression to predict churn.

4. **Evaluate Thresholds**  
   Computes precision, recall, F1-score for all thresholds; saves plots in `/plots`.

5. **Pick Best Threshold**  
   Finds the threshold with the highest F1-score for business action.

## Results and Insights This technique can help in identifying risky customers more effectively.

- Churn prediction is challenging due to class imbalance.
- Threshold tuning can improve recall for identifying risky customers.
- See `ML_Project_Report-1.pdf` and `Threshold-Tuning-for-Telco-Customer-Churn-Prediction.pptx` for full results.
