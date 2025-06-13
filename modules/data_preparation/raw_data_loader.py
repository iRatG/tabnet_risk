import pandas as pd

def load_raw_data(path: str, format: str = "csv") -> pd.DataFrame:
    if format == "csv":
        df = pd.read_csv(path)
    elif format == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file format: use 'csv' or 'parquet'")
    for col in df.columns:
        if 'date' in col.lower():
            df[col] = df[col].astype(str)
    return df
