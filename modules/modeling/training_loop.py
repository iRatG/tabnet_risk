import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
import pickle
from modules.modeling.custom_tabnet_classifier import TabNetClassifier

def train_tabnet_model(df: pd.DataFrame, target_column: str = "bucket_future"):
    df = df.copy()
    for col in df.columns:
        if 'date' in col.lower():
            df[col] = df[col].astype(str)

    X = df.drop(columns=[target_column, "loan_id"]).values
    le = LabelEncoder()
    y = le.fit_transform(df[target_column].values)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
    valid_data = TensorDataset(torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid))

    train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=1024)

    model = TabNetClassifier(input_dim=X.shape[1], output_dim=len(le.classes_))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model = None

    for epoch in range(50):
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += loss_fn(preds, yb).item()
                all_preds.extend(preds.argmax(dim=1).cpu().numpy())
                all_targets.extend(yb.cpu().numpy())

        val_loss /= len(valid_loader)
        print(f"Epoch {epoch+1}: Train Loss={running_loss:.4f}, Valid Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

    model.load_state_dict(best_model)

    with open("tabnet_model.pkl", "wb") as f:
        pickle.dump({
            "model_state_dict": model.state_dict(),
            "label_encoder": le,
            "scaler": scaler,
            "input_dim": X.shape[1],
            "output_dim": len(le.classes_)
        }, f)
