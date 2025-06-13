# Сохраняем скрипт main.py для автоматизации запуска пайплайна
main_py_code = """
import os
import pandas as pd
import subprocess

# Шаг 1: Проверка наличия исходных данных
data_path = "synthetic_credit_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Файл {data_path} не найден. Убедитесь, что данные синтезированы.")

# Шаг 2: Генерация финальной таблицы
from datetime import datetime

def generate_target(df, reference_date):
    ref_df = df[df["report_date"] == reference_date].copy()
    ref_df["bucket_future"] = pd.cut(ref_df["days_past_due"],
                                      bins=[-1, 0, 30, 60, 90, float("inf")],
                                      labels=[0, 1, 2, 3, 4])
    return ref_df[["loan_id", "bucket_future"]]

def calculate_10d_aggregates(df):
    grouped = df.groupby("loan_id")
    result = grouped.agg({
        "principal_outstanding": ["mean", "max", "min"],
        "interest_outstanding": ["mean", "max", "min"],
        "principal_overdue": ["mean", "max"],
        "interest_overdue": ["mean", "max"],
        "days_past_due": ["mean", "max"],
        "report_date": "count"
    })
    result.columns = ["_".join(col) for col in result.columns]
    result = result.reset_index()
    return result

def add_history_aggregates(df):
    df["month"] = pd.to_datetime(df["report_date"]).dt.to_period("M")
    grouped = df.groupby(["loan_id", "month"])
    agg = grouped.agg({
        "principal_outstanding": "mean",
        "interest_outstanding": "mean",
        "days_past_due": "mean"
    }).reset_index()
    agg.columns = ["loan_id", "month", "po_mean_90d", "io_mean_90d", "dpd_mean_90d"]
    return agg

print("📥 Загрузка данных")
df_raw = pd.read_csv(data_path)
df_raw["report_date"] = pd.to_datetime(df_raw["report_date"]).astype(str)

print("🎯 Генерация таргета")
df_target = generate_target(df_raw, "2025-05-10")

print("🔄 Агрегации")
df_10d = calculate_10d_aggregates(df_raw)
df_hist = add_history_aggregates(df_raw)

print("🧩 Объединение")
df_merged = pd.merge(df_10d, df_hist, on="loan_id", how="inner")
df_final = pd.merge(df_merged, df_target, on="loan_id", how="inner")
df_final.to_csv("final_dataset_for_training.csv", index=False)
print("✅ Финальный датасет сохранён: final_dataset_for_training.csv")

# Шаг 3: Запуск обучения
print("🚀 Запуск обучения TabNet")
subprocess.run(["python", "train_tabnet_cpu.py"])
"""

main_script_path = "/mnt/data/main.py"
with open(main_script_path, "w") as f:
    f.write(main_py_code)

main_script_path
