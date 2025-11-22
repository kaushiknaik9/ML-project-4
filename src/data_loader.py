import pandas as pd

def load_telco_data(path):
#    Loads the Telco Customer Churn dataset.
    df = pd.read_csv(path)
    
    # Clean TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    
    # Encode target
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    
    return df