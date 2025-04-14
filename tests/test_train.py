import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.fc = nn.Linear(4, 3)  # 4 input features, 3 output classes

    def forward(self, x):
        return self.fc(x)

def test_model_training():
    # Load training data
    df = pd.read_csv("data/train.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = IrisModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0, "Training loss should be positive"
        break  # Test one step only

    print("Training test passed!")
