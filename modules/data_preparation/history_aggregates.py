import pandas as pd

def add_history_aggregates(df: pd.DataFrame, days_list: list = [90, 180]) -> pd.DataFrame:
    df = df.copy()
    df["report_date"] = pd.to_datetime(df["report_date"])
    df.set_index("report_date", inplace=True)
    results = []

    for days in days_list:
        temp = df.groupby("loan_id").rolling(f"{days}D", min_periods=1).agg({
            "principal_outstanding": "mean",
            "principal_overdue": "mean",
            "interest_overdue": "mean",
            "days_past_due": "mean"
        }).reset_index()

        temp = temp.rename(columns={
            "principal_outstanding": f"po_mean_{days}d",
            "principal_overdue": f"pod_mean_{days}d",
            "interest_overdue": f"iod_mean_{days}d",
            "days_past_due": f"dpd_mean_{days}d"
        })

        results.append(temp)

    merged = results[0]
    for more in results[1:]:
        merged = pd.merge(merged, more, on=["loan_id", "report_date"], how="outer")

    df.reset_index(inplace=True)
    final = pd.merge(df, merged, on=["loan_id", "report_date"], how="left")
    return final
