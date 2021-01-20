import torch.nn as nn

class GRU_classifier(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout): # add arguments as you need them
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout)  
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)  

        
    def forward(self, batch, device):
        gru, h_0 = self.gru(batch)
        out = self.fc(gru).to(device)
        output = self.softmax(out)
        
        return output
