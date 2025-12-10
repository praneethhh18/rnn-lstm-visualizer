import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        
    def forward(self, x):
        out, h_n = self.rnn(x)
        return out, h_n


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return out, h_n, c_n
