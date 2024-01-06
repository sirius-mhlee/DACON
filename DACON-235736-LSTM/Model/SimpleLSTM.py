import torch
import torch.nn as nn

from Model.Layer import CustomLinearLayer

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size, 64, 2, batch_first=True)

        self.regressor = CustomLinearLayer(64, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(x.device) 
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)

        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = self.regressor(x[:, -1, :])
        return x
