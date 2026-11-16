from torch import nn

class NeuralReceiver(nn.Module):
    def __init__(self, hidden_dim = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim,4),
        )

    def forward(self,features):
        return self.network(features)