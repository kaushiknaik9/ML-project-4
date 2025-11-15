import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Encodes categorical columns and scales numeric columns.
    Returns X, y and fitted scaler.
    """

    y = df["Churn"]
    X = df.drop(["Churn", "customerID"], axis=1)

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns

    # One-hot encode categorical columns
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Scale numeric columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y, scaler
