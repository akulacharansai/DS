import torch
import torch.nn as nn
import pandas as pd


class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.fc = nn.Linear(4, 3)  # 4 input features, 3 output classes

    def forward(self, x):
        return self.fc(x)

def test_inference_predictions():
    # Load inference data
    df = pd.read_csv("data/inference.csv")
    X = df.iloc[:, :].values
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Load the trained model
    model = IrisModel()
    model.load_state_dict(torch.load("model/model.pth"))
    model.eval()

    predictions = model(X_tensor).argmax(dim=1).numpy()

    assert len(predictions) > 0, "No predictions made"
    print("Inference test passed!")