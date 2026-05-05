import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        # Định nghĩa các tầng (layers)
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Luồng dữ liệu đi qua mạng nơ-ron
        x = F.relu(self.fc1(x)) # Kích hoạt phi tuyến tính
        x = self.fc2(x)
        return x