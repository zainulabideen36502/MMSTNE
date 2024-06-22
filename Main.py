import torch
import torch.nn as nn
import torch.nn.functional as F

# NLSTM with Highway Connections
class NLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.highway = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        highway_out = F.relu(self.highway(lstm_out))
        out = lstm_out + highway_out
        return out

# Adaptive Channel Encoder (ACE) Block
class ACEBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ACEBlock, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1)
        self.transition = nn.Linear(output_channels, output_channels)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.transition(x))
        return x

# Full Model
class ProposedModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ProposedModel, self).__init__()
        self.nlstm = NLSTM(input_size, hidden_size)
        self.ace = ACEBlock(input_channels=hidden_size, output_channels=hidden_size)
        self.decoder = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.nlstm(x)
        x = x.unsqueeze(1)  # Adjust for 3D conv input
        x = self.ace(x)
        x = self.decoder(x)
        return x

# Example Usage
if __name__ == "__main__":
    input_size = 10
    hidden_size = 50
    num_classes = 1

    model = ProposedModel(input_size, hidden_size, num_classes)
    sample_input = torch.randn(32, 10, input_size)  # Batch size of 32, sequence length of 10
    output = model(sample_input)
    print(output.shape)

