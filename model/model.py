from base import BaseModel
import torch.nn as nn
import torch.nn.functional as F


class MnistModel(BaseModel):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.gru = nn.GRU()
        
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
