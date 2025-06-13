import pandas as pd

def calculate_10d_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["report_date"] = pd.to_datetime(df["report_date"])
    df["month"] = df["report_date"].dt.to_period("M")
    df = df[df["report_date"].dt.day <= 10]

    grouped = df.groupby(["loan_id", "month"]).agg({
        "principal_outstanding": ["mean", "max", "min"],
        "interest_outstanding": ["mean", "max"],
        "principal_overdue": ["mean", "max"],
        "interest_overdue": ["mean", "max"],
        "days_past_due": ["mean", "max"],
        "report_date": "count"
    })

    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    return grouped
