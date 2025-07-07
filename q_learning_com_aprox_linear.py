import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 7 classes + "unknown" prediction
reward_matrix=[[2,  -2,-3,	-3, -2,	-3,	-3,	-1],
               [-2, 3,	-4,	-4, -2,	-4,	-4,	-1],
               [-2, -2, 1,	-2, -3,	-2,	-2,	-1],
               [-2, -2, -2,	1,  -3,	-2,	-2,	-1],
               [-4, -3, -5,	-5, 5,  -5,	-5,	-1],
               [-2, -2, -2,	-2, -3, 1,	-2,	-1],
               [-2, -2, -2,	-2,	-3,	-2,	1,	-1]]
reward_matrix = torch.tensor(reward_matrix).float()

X = np.load("nmed_rn34_ham10k_vectors.npy")
csv = pd.read_csv("vectorDB3.csv")
cols=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc', 'dx', 'unknown']
col_to_idx = {col: idx for idx, col in enumerate(cols)}
y = np.array([col_to_idx[label] for label in csv["dx"]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=727)

dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train))
loader = DataLoader(dataset, batch_size=1024, shuffle=True)

input_dim = X.shape[1]
num_classes = reward_matrix.shape[1]

class LinearQ(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)  # (batch_size, num_classes)

model = LinearQ(input_dim, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 10000
avg_losses=[]
for epoch in range(num_epochs):
    total_loss = 0
    for batch_x, batch_y in loader:
        q_values = model(batch_x)  # (B, C)

        classes = torch.argmax(q_values, dim=1) # Greedy works best
        rewards = reward_matrix[batch_y, classes]

        q_pred = q_values[torch.arange(len(batch_x)), classes]
        loss = nn.MSELoss()(q_pred, rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    
    avg_loss = total_loss / len(dataset)
    avg_losses.append(avg_loss)
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

with open("q_losses2.txt","w") as f:
    f.write(",".join(map(str,avg_losses)))

model.eval()
X_test_tensor = torch.from_numpy(X_test).float()

with torch.no_grad():
    q_test = model(X_test_tensor)  # (N_test, num_classes)
    predicted_classes = torch.argmax(q_test, dim=1)

y_test_tensor = torch.from_numpy(y_test)
rewards = reward_matrix[y_test_tensor, predicted_classes]
average_reward = rewards.float().mean().item()

print(f"Average Test Reward: {average_reward:.4f}")

output_df = pd.DataFrame({
    'actual': [cols[y] for y in y_test],
    'predicted': [cols[y] for y in predicted_classes.numpy()],
})
output_df.to_csv("q_preds2.csv", index=False)