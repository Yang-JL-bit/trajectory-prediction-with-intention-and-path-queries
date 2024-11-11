import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size = 3) -> None:
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm_layer = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dtype=torch.float64)
        self.output_layer = nn.Linear(hidden_size, output_size, dtype=torch.float64)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float64)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float64)
        out, _ = self.lstm_layer(x, (h0, c0))
        # out = F.softmax(self.output_layer(out[:, -1, :]), dim=-1)
        return out[:, -1, :]
