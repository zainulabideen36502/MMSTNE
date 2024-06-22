
import torch
import torch.nn as nn
import torch.optim as optim
class NLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(NLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.highway = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        highway_out = self.highway(out)
        return highway_out + out

class ACEBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ACEBlock, self).__init__()
        self.embedding = nn.Conv3d(input_channels, output_channels, kernel_size=(1, 1, 1))
        self.local_cnn = nn.Conv3d(output_channels, output_channels, kernel_size=(3, 3, 3), padding=1)
        self.adaptive_channel_encoder = nn.Conv3d(output_channels, output_channels, kernel_size=(3, 3, 3), padding=1)
        self.adaptive_transition = nn.Conv3d(output_channels, output_channels, kernel_size=(3, 3, 3), padding=1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.local_cnn(x)
        x = self.adaptive_channel_encoder(x)
        x = self.adaptive_transition(x)
        return x

class MMSTNE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, input_channels, output_channels):
        super(MMSTNE, self).__init__()
        self.nlstm = NLSTM(input_size, hidden_size, num_layers)
        self.ace_block = ACEBlock(input_channels, output_channels)
        self.decoder = nn.Linear(hidden_size, input_size)
        
    def forward(self, x, external_factors):
        nlstm_out = self.nlstm(x)
        ace_out = self.ace_block(external_factors)
        ace_out_flat = ace_out.view(nlstm_out.size(0), nlstm_out.size(1), -1)
        combined = nlstm_out + ace_out_flat
        output = self.decoder(combined)
        return output
