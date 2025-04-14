import torch
import pandas as pd
import torch.nn as nn

# Define the Iris Model
class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.fc = nn.Linear(4, 3)  # 4 input features, 3 output classes

    def forward(self, x):
        return self.fc(x)

# Load inference data
df = pd.read_csv("data/inference.csv")
print("Inference Dataframe shape:", df.shape)  # Debug: Check number of columns
print("Columns:", df.columns.tolist())  # Debug: List column names

# Use all columns as features since no label is present
X = df.values  # All 4 columns as features
print("Feature shape:", X.shape)  # Debug: Check feature dimensions

# Validate feature count
expected_features = 4
if X.shape[1] != expected_features:
    raise ValueError(f"Expected {expected_features} features in inference.csv, but got {X.shape[1]} columns. Check your data or adjust the model.")

# Convert to PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)

# Load the trained model
model = IrisModel()
model.load_state_dict(torch.load("model/model.pth"))
model.eval()

# Make predictions
with torch.no_grad():  # Disable gradient computation for inference
    predictions = model(X_tensor).argmax(dim=1).numpy()

# Save predictions to a CSV file
output_df = pd.DataFrame({"Predicted": predictions})
output_df.to_csv("predictions.csv", index=False)

print("Inference complete! Predictions saved as 'predictions.csv'.")