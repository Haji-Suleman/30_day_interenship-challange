
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from sklearn.preprocessing import StandardScaler


train_df = pd.read_csv(
    "C:\\Users\\Haji Suleman\\Desktop\\30 day interenship\\titanic\\train.csv"
)
test_df = pd.read_csv(
    "C:\\Users\\Haji Suleman\\Desktop\\30 day interenship\\titanic\\test.csv"
)

# =========================
# FEATURE ENGINEERING
# =========================

# Title extraction
train_df["Title"] = train_df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
test_df["Title"] = test_df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

rare_titles = [
    "Lady",
    "Countess",
    "Capt",
    "Col",
    "Don",
    "Dr",
    "Major",
    "Rev",
    "Sir",
    "Jonkheer",
    "Dona",
]

train_df["Title"] = train_df["Title"].replace(rare_titles, "Rare")
test_df["Title"] = test_df["Title"].replace(rare_titles, "Rare")

train_df["Title"] = train_df["Title"].replace(
    {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
)
test_df["Title"] = test_df["Title"].replace(
    {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
)

# Encode Sex
train_df["Sex"] = train_df["Sex"].map({"female": 0, "male": 1})
test_df["Sex"] = test_df["Sex"].map({"female": 0, "male": 1})

# Family features
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"]
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"]

train_df["IsAlone"] = (train_df["FamilySize"] == 0).astype(int)
test_df["IsAlone"] = (test_df["FamilySize"] == 0).astype(int)

# Fare per person
train_df["FarePerPerson"] = train_df["Fare"] / (train_df["FamilySize"] + 1)
test_df["FarePerPerson"] = test_df["Fare"] / (test_df["FamilySize"] + 1)

# Drop useless columns
train_df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"], inplace=True)
test_passenger_id = test_df["PassengerId"]
test_df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"], inplace=True)

# One-hot encoding
train_df = pd.get_dummies(train_df, columns=["Embarked", "Title"], drop_first=True)
test_df = pd.get_dummies(test_df, columns=["Embarked", "Title"], drop_first=True)

# Align columns (CRITICAL)
train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

# =========================
# HANDLE MISSING VALUES
# =========================

train_df["Age"].fillna(train_df["Age"].median(), inplace=True)
test_df["Age"].fillna(test_df["Age"].median(), inplace=True)

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

train_df = train_df.astype(float)
test_df = test_df.astype(float)

# =========================
# SPLIT DATA
# =========================

X = train_df.drop(columns=["Survived"])
y = train_df["Survived"]

X_np = X.values
y_np = y.values

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42
)

# =========================
# SCALING (NO LEAKAGE)
# =========================

scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_np)
X_test_np = scaler.transform(X_test_np)

# Convert to tensor
X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32)

# =========================
# MODEL
# =========================


class TitanicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(X_train.shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.model(x)



torch.manual_seed(42)
model = TitanicModel()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# TRAINING
# =========================

epochs = 300

for epoch in range(epochs):
    model.train()

    logits = model(X_train)
    loss = loss_fn(logits, y_train.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test)
        probs = torch.sigmoid(test_logits)
        preds = (probs.squeeze() >= 0.5).float()

        acc = (preds == y_test).float().mean()
        test_loss = loss_fn(test_logits, y_test.unsqueeze(1))

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch} | Loss {loss:.4f} | Test Loss {test_loss:.4f} | Acc {acc:.4f}"
        )

# =========================
# PREDICTION
# =========================

test_scaled = scaler.transform(test_df.values)
test_tensor = torch.tensor(test_scaled, dtype=torch.float32)

with torch.inference_mode():
    test_logits = model(test_tensor)
    probs = torch.sigmoid(test_logits).squeeze()
    predictions = (probs >= 0.5).int().numpy()

submission = pd.DataFrame(
    {"PassengerId": test_passenger_id.astype(int), "Survived": predictions}
)

submission.to_csv("submission.csv", index=False)
