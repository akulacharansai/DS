import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

# Define the Iris Model
class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.fc = nn.Linear(4, 3)  # 4 input features, 3 output classes

    def forward(self, x):
        return self.fc(x)

# Load training data
df = pd.read_csv("data/train.csv")
print("Training Dataframe shape:", df.shape)
X = df.iloc[:, :-1].values  # Features (should be 4 columns)
y = df.iloc[:, -1].values   # Labels

if X.shape[1] != 4:
    raise ValueError(f"Expected 4 features in train.csv, but got {X.shape[1]} columns")

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model, loss, and optimizer
model = IrisModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())
    
    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    
    class_report = classification_report(all_labels, all_preds, target_names=['setosa', 'versicolor', 'virginica'], zero_division=0)
    print("Classification Report:")
    print(class_report)

# Save the trained model
save_path = os.path.join(os.getcwd(), 'model')
if not os.path.exists(save_path):
    os.makedirs(save_path)

torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
print(f"Model saved at: {os.path.join(save_path, 'model.pth')}")

# Load inference data for evaluation
inference_df = pd.read_csv("data/inference.csv")
print("Inference Dataframe shape:", inference_df.shape)
X_inf = inference_df.iloc[:, :-1].values
y_inf = inference_df.iloc[:, -1].values

if X_inf.shape[1] != 4:
    raise ValueError(f"Expected 4 features in inference.csv, but got {X_inf.shape[1]} columns")

X_inf_tensor = torch.tensor(X_inf, dtype=torch.float32)
y_inf_tensor = torch.tensor(y_inf, dtype=torch.long)

model.eval()
with torch.no_grad():
    outputs_inf = model(X_inf_tensor)
    _, predicted_inf = torch.max(outputs_inf.data, 1)
    inf_accuracy = accuracy_score(y_inf, predicted_inf.numpy())
    inf_report = classification_report(y_inf, predicted_inf.numpy(), target_names=['setosa', 'versicolor', 'virginica'], zero_division=0)

print("Inference Set Results:")
print(f"Accuracy: {inf_accuracy:.4f}")
print("Classification Report:")
print(inf_report)

predictions_df = pd.DataFrame({'Predicted': predicted_inf.numpy(), 'Actual': y_inf})
predictions_df.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")