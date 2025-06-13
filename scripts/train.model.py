# Entry-point for training pipeline
from modules.data_preparation.final_dataset_builder import build_final_dataset
from modules.modeling.custom_tabnet_classifier import train_tabnet_model

if __name__ == '__main__':
    df = build_final_dataset("data/synthetic_credit_data.csv", "2025-05-31")
    train_tabnet_model(df)
