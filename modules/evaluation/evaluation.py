import pickle
import torch
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def evaluate_model(model_path: str, data: pd.DataFrame, target_column: str = "bucket_future"):
    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    model_state_dict = obj["model_state_dict"]
    label_encoder = obj["label_encoder"]
    scaler = obj["scaler"]

    X = data.drop(columns=[target_column, "loan_id"])
    y = label_encoder.transform(data[target_column].values)
    X_scaled = scaler.transform(X.values)

    model = torch.load("model_architecture.pt")  # загружается отдельно, если нужно
    model.load_state_dict(model_state_dict)
    model.eval()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_tensor).argmax(dim=1).numpy()

    print("Classification Report:")
    print(classification_report(y, preds, target_names=label_encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y, preds))
