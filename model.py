import torch
import torch.nn as nn

class IntentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, num_classes):
        super(IntentClassifier, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)      # [batch_size, seq_len, embedding_dim]
        _, (hidden, _) = self.rnn(embedded)
        # hidden[-1]: [batch_size, hidden_size]
        output = self.fc(hidden[-1])      # [batch_size, num_classes]
        return output
