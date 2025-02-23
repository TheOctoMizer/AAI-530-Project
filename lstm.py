import torch
import torch.nn as nn

class TrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(TrafficLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Add batch normalization for input
        self.batch_norm = nn.BatchNorm1d(input_size)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout,
                           bidirectional=True)
        
        # Double the hidden size due to bidirectional LSTM
        lstm_output_size = hidden_size * 2
        
        # Add deeper fully connected layers with batch norm and dropout
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.bn1 = nn.BatchNorm1d(lstm_output_size // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Separate layer paths for classification and regression
        self.fc_class = nn.Linear(lstm_output_size // 2, lstm_output_size // 4)
        self.fc_reg = nn.Linear(lstm_output_size // 2, lstm_output_size // 4)
        
        self.fc_classification = nn.Linear(lstm_output_size // 4, num_classes)
        self.fc_regression = nn.Linear(lstm_output_size // 4, 1)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Apply batch normalization to input
        x = x.squeeze(1)  # Remove sequence dimension temporarily
        x = self.batch_norm(x)
        x = x.unsqueeze(1)  # Restore sequence dimension
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the last time step
        
        # Common layers
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        # Separate paths
        class_out = self.fc_class(out)
        class_out = self.relu(class_out)
        class_out = self.fc_classification(class_out)
        
        reg_out = self.fc_reg(out)
        reg_out = self.relu(reg_out)
        reg_out = self.fc_regression(reg_out)
        
        return class_out, reg_out
