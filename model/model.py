from base import BaseModel
import torch.nn as nn
import torch.nn.functional as F


class GRU4REC(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh'):
        super(GRU4REC, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, nonlinearity)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        x, h_n = self.gru(x)
        data, batches = x
        out = (self.fc(data), batches)
        return out, h_n
