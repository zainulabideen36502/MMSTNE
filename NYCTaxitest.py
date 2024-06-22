import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

def load_nyctaxi_data(file_path):
    # Load the dataset (assuming it's in CSV format for this example)
    data = pd.read_csv(file_path)
    return data

def normalize_data(data):
    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler

def generate_spatio_temporal_features(data, seq_length=10):
    # Generate sequences of spatio-temporal features
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

def split_data(sequences):
    # Split the data into training, validation, and test sets
    train, test = train_test_split(sequences, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    return train, val, test

# Preprocessing pipeline
file_path = 'path_to_nyctaxi_data.csv'  # Update with your file path
data = load_nyctaxi_data(file_path)

# Example: assuming 'flow' column contains the crowd flow data
crowd_flow_data = data['flow'].values.reshape(-1, 1)  # Reshape for normalization

# Normalize the data
normalized_data, scaler = normalize_data(crowd_flow_data)

# Generate spatio-temporal features
seq_length = 10  # Example sequence length
sequences = generate_spatio_temporal_features(normalized_data, seq_length)

# Split the data
train_data, val_data, test_data = split_data(sequences)

# Example: Creating dummy external factors (you need actual external factors data)
external_factors_train = np.random.randn(len(train_data), 1, 10, 10, 10)
external_factors_val = np.random.randn(len(val_data), 1, 10, 10, 10)
external_factors_test = np.random.randn(len(test_data), 1, 10, 10, 10)

# Save or use the preprocessed data for training the model
# Example: Using the preprocessed data with the model
train_inputs = torch.tensor(train_data[:, :-1], dtype=torch.float32)
train_targets = torch.tensor(train_data[:, -1], dtype=torch.float32)
val_inputs = torch.tensor(val_data[:, :-1], dtype=torch.float32)
val_targets = torch.tensor(val_data[:, -1], dtype=torch.float32)
test_inputs = torch.tensor(test_data[:, :-1], dtype=torch.float32)
test_targets = torch.tensor(test_data[:, -1], dtype=torch.float32)

external_factors_train = torch.tensor(external_factors_train, dtype=torch.float32)
external_factors_val = torch.tensor(external_factors_val, dtype=torch.float32)
external_factors_test = torch.tensor(external_factors_test, dtype=torch.float32)

# Example: Running the model with preprocessed data
model = MMSTNE(input_size=10, hidden_size=20, num_layers=2, input_channels=1, output_channels=3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_inputs, external_factors_train)
    loss = criterion(outputs, train_targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Validation (simplified)
model.eval()
with torch.no_grad():
    val_outputs = model(val_inputs, external_factors_val)
    val_loss = criterion(val_outputs, val_targets)
    print(f"Validation Loss: {val_loss.item()}")
```


1. Load the Dataset:
   - Load the NYCTaxi dataset using Pandas.

2. Normalize the Data:
   - Use `MinMaxScaler` to normalize the crowd flow data.

3. Generate Spatio-Temporal Features:
   - Create sequences of spatio-temporal data with a specified sequence length (e.g., 10).

4. Split the Data:
   - Split the data into training, validation, and test sets using `train_test_split`.

5. External Factors:
   - For simplicity, dummy data is used for external factors. Replace this with actual external factors data.

6. Prepare Tensors:
   - Convert the preprocessed data into PyTorch tensors.

7. Training Loop:
   - A simplified training loop is provided to illustrate how to train the model with the preprocessed data.

### Important Notes:
- Replace the `file_path` with the actual path to your NYCTaxi dataset.
- Ensure the column names and the format match your dataset.
- Replace the dummy external factors with actual external factors.
- Adjust hyperparameters (e.g., `input_size`, `hidden_size`, `num_layers`, `input_channels`, `output_channels`, learning rate) based on your specific requirements.
