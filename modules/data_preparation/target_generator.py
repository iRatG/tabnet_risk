import pandas as pd

def generate_target(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    df["report_date"] = pd.to_datetime(df["report_date"])
    target_dt = pd.to_datetime(target_date)
    df_target = df[df["report_date"] == target_dt].copy()

    def bucketize(days: int) -> int:
        if pd.isna(days): return 0
        if days == 0: return 0
        elif days <= 30: return 1
        elif days <= 60: return 2
        elif days <= 90: return 3
        else: return 4

    df_target["bucket_future"] = df_target["days_past_due"].apply(bucketize)
    return df_target[["loan_id", "bucket_future"]]
