
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# ---------------- TabNet Components ----------------
class GLU_Block(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.fc_gate = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.fc_gate(x))

class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.block1 = GLU_Block(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.block2 = GLU_Block(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.bn1(self.block1(x))
        x = self.bn2(self.block2(x))
        return x

class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x, prior):
        x = self.bn(self.fc(x))
        x = F.softmax(x * prior, dim=-1)
        return x

class TabNetClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, n_d=32, n_steps=3):
        super().__init__()
        self.initial_transform = FeatureTransformer(input_dim, n_d)
        self.steps = nn.ModuleList()
        for _ in range(n_steps):
            transformer = FeatureTransformer(input_dim, n_d)
            attention = AttentiveTransformer(n_d, input_dim)
            self.steps.append(nn.ModuleDict({"transformer": transformer, "attention": attention}))
        self.classifier = nn.Linear(n_d, output_dim)

    def forward(self, x):
        prior = torch.ones(x.shape).to(x.device)
        out = self.initial_transform(x)
        aggregated = 0
        for step in self.steps:
            mask = step["attention"](out, prior)
            x_step = x * mask
            out = step["transformer"](x_step)
            aggregated += out
            prior *= (1 - mask)
        logits = self.classifier(aggregated)
        return logits

# ---------------- Data Loading ----------------
df = pd.read_csv("final_dataset_for_training.csv")
X = df.drop(columns=["loan_id", "bucket_future", "month"]).values
y = LabelEncoder().fit_transform(df["bucket_future"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
valid_ds = TensorDataset(torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid))
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=4)

# ---------------- Training Loop ----------------
device = torch.device("cpu")
model = TabNetClassifier(input_dim=X.shape[1], output_dim=len(set(y)))
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

train_losses, valid_losses = [], []

for epoch in range(15):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_loader))

    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for xb, yb in valid_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            valid_loss += loss_fn(preds, yb).item()
    valid_losses.append(valid_loss / len(valid_loader))

    print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Valid Loss = {valid_losses[-1]:.4f}")

# ---------------- Visualization ----------------
plt.plot(train_losses, label="Train Loss")
plt.plot(valid_losses, label="Valid Loss")
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")

# ---------------- Save Model ----------------
with open("tabnet_synthetic_model.pkl", "wb") as f:
    pickle.dump(model.state_dict(), f)
