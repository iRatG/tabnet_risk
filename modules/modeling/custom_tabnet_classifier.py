import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, input_dim, output_dim, n_d=64, n_steps=5):
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
